"""LLMClient ABC and shared dataclasses.

This module is the **Mac vs RTX 5090 seam**. Concrete implementations
(`OllamaClient`, `SGLangClient`, `MockClient`, future `VLLMClient`) all
return the same Completion / Choice shapes so the rest of the harness
is platform-agnostic.

Two methods cover the proposal's two-pass design:

- ``generate(...)`` — Pass 1 free reasoning. ``max_tokens`` *is* the budget B.
- ``generate_choice(...)`` — Pass 2 constrained move selection. SGLang exposes
  a native ``choices`` operator; backends without one (e.g. Ollama) approximate
  via per-choice scoring (see ``OllamaClient``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class Completion:
    """Result of a free-form LLM generation call (Pass 1)."""

    text: str
    n_input_tokens: int
    n_output_tokens: int
    finish_reason: str  # "stop" | "length" | "error" | "tool" | ...
    latency_ms: float
    model: str
    raw: dict = field(default_factory=dict)  # backend-specific payload for debugging


@dataclass(slots=True)
class Choice:
    """Result of a constrained-choice call (Pass 2)."""

    choice_index: int
    choice_text: str
    logprobs: list[float] | None  # one per candidate, if the backend exposes them
    n_input_tokens: int
    n_output_tokens: int
    finish_reason: str  # "stop" | "length" | … (Ollama: done_reason)
    latency_ms: float
    model: str
    raw: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract LLM backend.

    All methods are async; a sequential harness can ``asyncio.run`` them.
    Backends should be safe to share across coroutines (httpx clients are).
    """

    name: str = "abstract"

    @abstractmethod
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
        """Pass 1: free reasoning. ``max_tokens`` is the budget B."""

    @abstractmethod
    async def generate_choice(
        self,
        prompt: str,
        *,
        choices: list[str],
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
    ) -> Choice:
        """Pass 2: constrained selection. Returns exactly one of ``choices``."""

    async def aclose(self) -> None:
        """Close any open transports."""
        return None
