"""Memory manager ABC.

The proposal compares two memory modes (Task 5):

- ``StructuredSummaryMemory`` — calls the agent's LLM to summarize history
  every ``summarize_every`` turns; renders ``<= max_summary_tokens`` tokens.
- ``FullHistoryMemory`` — concatenates the chronological move list verbatim.

Both expose the same interface so the LLM agent doesn't care which one is
in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    turn_idx: int
    player: int  # +1 / -1
    move_str: str
    state_after_serialized: dict[str, Any]


@dataclass
class MemorySnapshot:
    """Returned by ``MemoryManager.render()``; the runner stores it in ``summaries``."""

    text: str
    n_chars: int
    n_tokens_estimate: int
    kind: str  # "summary" | "full_history"
    extra: dict[str, Any] = field(default_factory=dict)


class MemoryManager(ABC):
    kind: str = "abstract"

    @abstractmethod
    def update(self, record: TurnRecord) -> None: ...

    @abstractmethod
    async def render(self) -> MemorySnapshot: ...

    def reset(self) -> None: ...
