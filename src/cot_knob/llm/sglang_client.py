"""SGLang HTTP client — primary backend for RTX 5080/5090 GPU inference.

Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint, which:
  - Automatically applies the model's chat template (critical for instruction
    models like Llama 3.1 8B Instruct).
  - Accepts SGLang's constrained-decoding extension via
    ``sampling_params.regex`` in the request body, eliminating parse_failed.
  - Returns standard OpenAI-format usage stats (prompt_tokens, completion_tokens).

Pass-2 ``generate_choice()`` also uses regex-constrained generation: the legal
move list is compiled into a regex alternation, so the model is forced to
output exactly one legal move string rather than free-text that we parse.

Constrained decoding notes
--------------------------
- ``regex`` must be a Python-compatible regex string (uses the ``outlines``
  library under the hood in SGLang).
- ``[\s\S]*`` at the start of a regex allows arbitrary reasoning prefix text
  before the constrained suffix.  This preserves the visible CoT trace while
  guaranteeing the move tag format.
- When ``regex`` is supplied, ``finish_reason`` is always ``"stop"`` (the regex
  was satisfied) unless ``max_tokens`` is hit first (``"length"``).
- ``supports_regex = True`` lets ``LLMAgent`` thread the Nim move regex through
  automatically — eliminating parse_failed as a confound.

RTX 5080 / sm_120 notes
-----------------------
- SGLang ≥ 0.3.0 ships sm_120 kernels via FlashInfer ≥ 0.1.9.
- Verify: ``python -c "import torch; print(torch.cuda.get_device_capability())"``
  should print (12, 0).
- If kernels are missing, add ``--attention-backend triton --disable-cuda-graph``
  to the server launch command.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from cot_knob.llm.client import Choice, Completion, LLMClient

DEFAULT_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
DEFAULT_MODEL = os.environ.get("SGLANG_MODEL", "default")


class SGLangClient(LLMClient):
    name = "sglang"
    supports_regex = True  # constrained decoding is natively supported

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── Pass 1: free reasoning (with optional constrained decoding) ───────────

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
        think: bool | None = None,  # ignored — Llama has no separate thinking field
        regex: str | None = None,
    ) -> Completion:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if seed is not None:
            body["seed"] = int(seed)
        if stop:
            body["stop"] = stop
        if regex is not None:
            # SGLang extends the OpenAI body with sampling_params for constrained decoding.
            body["sampling_params"] = {"regex": regex}

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish = choice.get("finish_reason") or "stop"
        usage = data.get("usage", {})
        return Completion(
            text=text,
            n_input_tokens=int(usage.get("prompt_tokens", 0)),
            n_output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=finish,
            latency_ms=latency_ms,
            model=data.get("model", self.model),
            raw=data,
        )

    # ── Pass 2: constrained choice selection ──────────────────────────────────

    async def generate_choice(
        self,
        prompt: str,
        *,
        choices: list[str],
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
    ) -> Choice:
        """Select one of ``choices`` via regex-constrained generation.

        Builds a regex that forces the model to output exactly one of the
        candidate strings, so no post-hoc parsing is needed.
        """
        if not choices:
            raise ValueError("SGLangClient.generate_choice: choices is empty")

        # Escape each choice for regex, then join as alternation.
        pattern = "(" + "|".join(re.escape(c) for c in choices) + ")"
        comp = await self.generate(
            prompt,
            max_tokens=max(len(c) for c in choices) + 4,  # tight budget
            temperature=temperature,
            seed=seed,
            system=system,
            regex=pattern,
        )
        text = comp.text.strip()
        try:
            idx = choices.index(text)
        except ValueError:
            # Fallback: find best partial match (should be rare with regex enforcement).
            # Flag it so the runner can record a parse-failure, mirroring OllamaClient.
            idx = next((i for i, c in enumerate(choices) if c in text), 0)
            comp.raw["__choice_parse_failed"] = True
        return Choice(
            choice_index=idx,
            choice_text=choices[idx],
            logprobs=None,
            n_input_tokens=comp.n_input_tokens,
            n_output_tokens=comp.n_output_tokens,
            finish_reason=comp.finish_reason,
            latency_ms=comp.latency_ms,
            model=comp.model,
            raw=comp.raw,
        )
