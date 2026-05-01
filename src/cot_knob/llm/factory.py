"""Factory: build an LLMClient from a config dict.

Centralizes the backend selection so the runner doesn't import any
backend-specific module directly.
"""

from __future__ import annotations

from typing import Any

from cot_knob.llm.client import LLMClient


def build_client(cfg: dict[str, Any]) -> LLMClient:
    """Build an LLMClient.

    cfg shape::

        {"backend": "ollama" | "sglang" | "mock",
         "name": "deepseek-r1:7b",
         "base_url": "http://localhost:11434",
         "temperature": 0.0,
         "extra": {...}}
    """
    backend = cfg.get("backend", "mock").lower()
    name = cfg.get("name") or cfg.get("model")
    extra = cfg.get("extra", {})

    if backend == "mock":
        from cot_knob.llm.mock_client import MockClient

        return MockClient(model=name or "mock-r1", **extra)
    if backend == "ollama":
        from cot_knob.llm.ollama_client import DEFAULT_MODEL, DEFAULT_URL, OllamaClient

        return OllamaClient(
            model=name or DEFAULT_MODEL,
            base_url=cfg.get("base_url") or DEFAULT_URL,
            **extra,
        )
    if backend == "sglang":
        from cot_knob.llm.sglang_client import DEFAULT_MODEL, DEFAULT_URL, SGLangClient

        return SGLangClient(
            model=name or DEFAULT_MODEL,
            base_url=cfg.get("base_url") or DEFAULT_URL,
            **extra,
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")
