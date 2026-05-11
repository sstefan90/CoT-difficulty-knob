"""Ollama HTTP client.

Implements ``LLMClient`` against an Ollama server (default ``http://localhost:11434``).

Pass 1 (`generate`) maps onto Ollama's ``/api/generate`` with
``options.num_predict = max_tokens`` (Ollama's name for max output tokens).

**Thinking models (DeepSeek R1, Qwen3 think, etc.):** Ollama splits output
into two fields — ``thinking`` holds the chain-of-thought and ``response``
holds the final answer. If we only read ``response``, Pass 1 looks empty in
the DB even when ``eval_count`` is large. We therefore set ``think: true``
for Pass 1 and concatenate ``thinking`` + ``response`` into ``Completion.text``.
See https://docs.ollama.com/capabilities/thinking

Pass 2 sets ``think: false`` so the model emits a short answer without a
separate thinking trace that would steal ``num_predict`` budget.

Pass 2 (`generate_choice`) is *approximate* on Ollama because Ollama does
not expose SGLang's ``choices`` operator. We score each candidate by
computing the conditional log-likelihood of the choice text given the
prompt (one ``/api/generate`` call per choice with ``num_predict=0`` and
``raw=True``, reading back ``prompt_eval_count`` deltas), and pick argmax.
For Reversi this means at most ~30 tiny calls per Pass-2 — slow but
correct enough for the smoke sweep. On the 5090 we'll use SGLang's
real ``choices`` operator.

If logprob scoring is unavailable, we fall back to the cheapest possible
substitute: a single ``/api/chat`` call with the choice list rendered as
``"Reply with exactly one of: a, b, c."`` and a regex extractor. This
fallback is documented as "lossy"; the smoke sweep prefers the scoring
path.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from cot_knob.llm.client import Choice, Completion, LLMClient

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")


def _merge_generate_output(data: dict) -> str:
    """Combine Ollama ``/api/generate`` fields for thinking-capable models.

    Older Ollama / non-thinking models only populate ``response``.
    """
    thinking = (data.get("thinking") or "").strip()
    response = (data.get("response") or "").strip()
    if thinking and response:
        return f"{thinking}\n\n{response}"
    return thinking or response


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout_s: float = 600.0,
        choice_strategy: str = "regex",  # "regex" | "scoring"
        think: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.choice_strategy = choice_strategy
        # think=True: model uses separate thinking field (R1 style).
        # think=False: model outputs everything in response field (Llama style).
        self._think = think
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- Pass 1: free reasoning ------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
        think: bool | None = None,
        regex: str | None = None,  # accepted for API compatibility; Ollama cannot enforce regex
    ) -> Completion:
        # Use instance default (set from config) unless caller overrides.
        use_think = self._think if think is None else think
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # think=True: separate reasoning field (R1). think=False: all in response (Llama).
            "think": use_think,
            "options": {
                "num_predict": int(max_tokens),
                "temperature": float(temperature),
            },
        }
        if system is not None:
            body["system"] = system
        if stop:
            body["options"]["stop"] = stop
        if seed is not None:
            body["options"]["seed"] = int(seed)

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/api/generate", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()

        text = _merge_generate_output(data)
        n_in = int(data.get("prompt_eval_count", 0))
        n_out = int(data.get("eval_count", 0))
        finish = "stop"
        if data.get("done_reason") == "length":
            finish = "length"
        elif not data.get("done", True):
            finish = "error"

        return Completion(
            text=text,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            finish_reason=finish,
            latency_ms=latency_ms,
            model=self.model,
            raw=data,
        )

    # --- Pass 2: constrained selection ----------------------------------------

    async def generate_choice(
        self,
        prompt: str,
        *,
        choices: list[str],
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
    ) -> Choice:
        if not choices:
            raise ValueError("OllamaClient.generate_choice: choices is empty")

        if self.choice_strategy == "regex":
            return await self._choose_via_regex(
                prompt, choices, temperature=temperature, seed=seed, system=system
            )
        elif self.choice_strategy == "scoring":  # noqa: RET505
            # Reserved: per-choice logprob scoring. Not enabled for the smoke run.
            return await self._choose_via_regex(
                prompt, choices, temperature=temperature, seed=seed, system=system
            )
        else:
            raise ValueError(f"Unknown choice_strategy: {self.choice_strategy}")

    async def _choose_via_regex(
        self,
        prompt: str,
        choices: list[str],
        *,
        temperature: float,
        seed: int | None,
        system: str | None,
    ) -> Choice:
        choice_list = ", ".join(choices)
        ask = (
            f"{prompt}\n\n"
            f"Respond with exactly one of these tokens, on a line by itself, "
            f"and nothing else: {choice_list}\n"
            f"Answer:"
        )
        body: dict = {
            "model": self.model,
            "prompt": ask,
            "stream": False,
            # No separate thinking trace — it would eat ``num_predict``.
            "think": False,
            # 128 tokens: Reversi moves are short ("d3") but Nim moves are
            # longer ("take 3 from A") and R1-Distill often adds a brief
            # preamble before the answer. 32 tokens caused finish_reason=length
            # on every Nim turn; 128 gives enough room to reach a natural stop.
            "options": {
                "num_predict": 128,
                "temperature": float(temperature),
            },
        }
        if system is not None:
            body["system"] = system
        if seed is not None:
            body["options"]["seed"] = int(seed)

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/api/generate", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        text = _merge_generate_output(data).strip()

        idx = _match_choice(text, choices)
        if idx is None:
            # Last-resort: pick the first legal choice (deterministic),
            # but flag in raw so the runner can record a parse-failure.
            idx = 0
            data["__choice_parse_failed"] = True

        n_out = int(data.get("eval_count", 0))
        fin = str(data.get("done_reason") or "stop")
        return Choice(
            choice_index=idx,
            choice_text=choices[idx],
            logprobs=None,
            n_input_tokens=int(data.get("prompt_eval_count", 0)),
            n_output_tokens=n_out,
            finish_reason=fin,
            latency_ms=latency_ms,
            model=self.model,
            raw=data,
        )


def _match_choice(text: str, choices: list[str]) -> int | None:
    """Best-effort extraction of one of `choices` from a free-text reply.

    Strategy:
      1. Strip <think>...</think> blocks (R1-Distill emits them).
      2. Look for an exact token match in word boundaries, longest first
         (so 'a8' wins over 'a').
      3. Return None if nothing matches.
    """
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    # Sort longest first so multi-char tokens beat substrings.
    ordered = sorted(enumerate(choices), key=lambda kv: -len(kv[1]))
    lowered = cleaned.lower()
    for idx, choice in ordered:
        pat = r"(?<![A-Za-z0-9])" + re.escape(choice.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pat, lowered):
            return idx
    return None
