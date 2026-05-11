"""Deterministic mock LLM client.

Used for unit tests and harness shakedown without GPU/Ollama. Pass 1 emits
a configurable number of "reasoning" tokens; Pass 2 selects a choice by a
deterministic hash of the prompt so games are reproducible per-seed.
"""

from __future__ import annotations

import hashlib
import time

from cot_knob.llm.client import Choice, Completion, LLMClient


def _stable_index(prompt: str, n: int, seed: int | None) -> int:
    h = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
    return int.from_bytes(h[:8], "big") % max(n, 1)


def _fake_reasoning(n_tokens: int, seed_str: str) -> str:
    """Produce text with ~n_tokens whitespace-separated tokens, deterministic per seed."""
    if n_tokens <= 0:
        return ""
    rng = hashlib.sha256(seed_str.encode()).digest()
    words = [
        "consider",
        "the",
        "corner",
        "control",
        "edge",
        "stability",
        "mobility",
        "frontier",
        "parity",
        "swing",
    ]
    out: list[str] = []
    i = 0
    while len(out) < n_tokens:
        out.append(words[rng[i % len(rng)] % len(words)])
        i += 1
    return " ".join(out)


class MockClient(LLMClient):
    name = "mock"

    def __init__(self, *, model: str = "mock-r1", latency_ms: float = 1.0) -> None:
        self.model = model
        self._latency_ms = latency_ms

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
        regex: str | None = None,  # accepted for API compatibility; ignored by mock
    ) -> Completion:
        text = _fake_reasoning(max_tokens, f"reason:{seed}:{prompt[:64]}")
        return Completion(
            text=text,
            n_input_tokens=max(1, len(prompt.split())),
            n_output_tokens=len(text.split()),
            finish_reason="stop" if max_tokens > 0 else "length",
            latency_ms=self._latency_ms,
            model=self.model,
            raw={"backend": "mock", "max_tokens": max_tokens},
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
            raise ValueError("MockClient.generate_choice: choices is empty")
        idx = _stable_index(prompt, len(choices), seed)
        t0 = time.perf_counter()
        # Trivially fast, but record a real latency anyway.
        latency = (time.perf_counter() - t0) * 1000.0 + self._latency_ms
        return Choice(
            choice_index=idx,
            choice_text=choices[idx],
            logprobs=None,
            n_input_tokens=max(1, len(prompt.split())),
            n_output_tokens=1,
            finish_reason="stop",
            latency_ms=latency,
            model=self.model,
            raw={"backend": "mock", "n_choices": len(choices)},
        )
