"""Agent ABC and shared dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cot_knob.games.base import GameState


@dataclass
class TurnTelemetry:
    """Per-turn record produced by an Agent.choose() call.

    The runner copies this into the SQLite ``turns`` and ``model_calls``
    tables. Optional fields are filled in only by certain agents
    (e.g. ``llm_pass1_text`` is empty for ``UCTAgent``).
    """

    chosen_move: int
    legal_moves: list[int]
    state_serialized: dict[str, Any]

    # LLM-only fields (empty for UCTAgent / FillerAgent).
    llm_pass1_text: str = ""
    llm_pass1_tokens_in: int = 0
    llm_pass1_tokens_out: int = 0
    llm_pass1_finish: str = ""
    llm_pass1_latency_ms: float = 0.0
    llm_pass2_choice_text: str = ""
    llm_pass2_latency_ms: float = 0.0
    llm_pass2_parse_failed: bool = False

    # UCT (or any MCTS) telemetry.
    uct_top3: list[dict[str, Any]] = field(default_factory=list)
    """List of {"move": str, "visits": int, "win_rate": float} entries."""

    extra: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for any game-playing agent."""

    name: str = "abstract"

    @abstractmethod
    async def choose(self, state: GameState, *, seed: int | None = None) -> TurnTelemetry:
        """Choose a move from ``state`` and return telemetry."""

    async def aclose(self) -> None:
        return None
