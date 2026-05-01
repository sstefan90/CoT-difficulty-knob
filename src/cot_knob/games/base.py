"""Game protocol shared by Ludii-backed and pure-Python game backends.

Concrete implementations:

- ``cot_knob.games.reversi`` — pure-Python Reversi (works on Mac today,
  no Java required). Used for the smoke sweep.
- ``cot_knob.games.ludii_reversi`` (future) — Ludii.jar via JPype, which
  is the proposal's primary game source. Same interface; swap in by
  changing one line in the runner.

The protocol is intentionally minimal so adding Avalon (proposal Task 13)
or any other Ludii game later is just another implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GameState(Protocol):
    """A single game state, immutable from the caller's perspective."""

    @property
    def turn_idx(self) -> int: ...

    @property
    def current_player(self) -> int:
        """1 or -1 (Black / White for Reversi)."""
        ...

    def is_terminal(self) -> bool: ...

    def winner(self) -> int | None:
        """+1, -1 for a winner; 0 for a draw; None if not terminal."""
        ...

    def legal_moves(self) -> list[int]:
        """List of legal move ids (e.g. flat board indices for Reversi)."""
        ...

    def apply_move(self, move: int) -> "GameState":
        """Return a new state after applying ``move``."""

    def to_serializable(self) -> dict[str, Any]:
        """JSON-serializable view of the state for prompts and logging."""

    def render_text(self) -> str:
        """Human-readable rendering. Used in prompts."""

    def move_to_str(self, move: int) -> str:
        """Render a move id (e.g. 26 -> 'c4')."""

    def str_to_move(self, s: str) -> int | None:
        """Parse a move string back to an id. None if unparseable."""
