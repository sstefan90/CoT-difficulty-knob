"""SGLang HTTP client (RTX 5090 production backend).

Stub implementation kept thin until we validate SGLang on sm_120. The
shape mirrors ``OllamaClient`` so the runner doesn't care which backend
is in use. When we wire this up on the 5090 box, the only changes are
inside this file.
"""

from __future__ import annotations

import os
import time

import httpx

from cot_knob.llm.client import Choice, Completion, LLMClient

DEFAULT_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
DEFAULT_MODEL = os.environ.get("SGLANG_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


class SGLangClient(LLMClient):
    name = "sglang"

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

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
    ) -> Completion:
        body = {
            "text": prompt if system is None else f"{system}\n\n{prompt}",
            "sampling_params": {
                "max_new_tokens": int(max_tokens),
                "temperature": float(temperature),
                "stop": stop or [],
            },
        }
        if seed is not None:
            body["sampling_params"]["seed"] = int(seed)

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/generate", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()

        text = data.get("text", "") if isinstance(data, dict) else str(data)
        meta = data.get("meta_info", {}) if isinstance(data, dict) else {}
        return Completion(
            text=text,
            n_input_tokens=int(meta.get("prompt_tokens", 0)),
            n_output_tokens=int(meta.get("completion_tokens", 0)),
            finish_reason=str(meta.get("finish_reason", "stop")),
            latency_ms=latency_ms,
            model=self.model,
            raw=data if isinstance(data, dict) else {"raw": data},
        )

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
            raise ValueError("SGLangClient.generate_choice: choices is empty")
        body = {
            "text": prompt if system is None else f"{system}\n\n{prompt}",
            "choices": choices,
        }
        if seed is not None:
            body["sampling_params"] = {"seed": int(seed), "temperature": float(temperature)}

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.base_url}/select", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()

        # SGLang returns the chosen text and per-choice logprobs.
        chosen_text = data.get("choice") if isinstance(data, dict) else None
        if chosen_text is None:
            chosen_text = data.get("text", choices[0]) if isinstance(data, dict) else choices[0]
        logprobs = data.get("choice_logprobs") if isinstance(data, dict) else None
        try:
            idx = choices.index(chosen_text)
        except ValueError:
            idx = 0
        return Choice(
            choice_index=idx,
            choice_text=choices[idx],
            logprobs=logprobs,
            n_input_tokens=0,
            latency_ms=latency_ms,
            model=self.model,
            raw=data if isinstance(data, dict) else {"raw": data},
        )
