"""Pure-Python Nim (normal play convention).

Rules: players alternate removing ≥1 stone from exactly one non-empty pile.
The player who takes the last stone wins (normal play).

Move encoding
-------------
  move_id = pile_index * (MAX_STONES + 1) + stones_taken
  MAX_STONES = 50  (generous ceiling; real games use far fewer)

This gives a unique non-negative integer per (pile, count) pair and keeps
move IDs compact enough for the runner's list indexing.

Pile labels: A, B, C, ... (index 0, 1, 2, ...)
Move strings: "take 2 from A", "take 1 from C", etc.
"""

from __future__ import annotations

import re
from functools import reduce
from operator import xor
from typing import Any

MAX_STONES: int = 50
_PILE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _encode(pile_idx: int, stones: int) -> int:
    return pile_idx * (MAX_STONES + 1) + stones


def _decode(move_id: int) -> tuple[int, int]:
    pile_idx, stones = divmod(move_id, MAX_STONES + 1)
    return pile_idx, stones


class NimState:
    """Immutable Nim game state implementing the GameState protocol."""

    def __init__(
        self,
        piles: tuple[int, ...],
        current_player: int = 1,
        turn_idx: int = 0,
    ) -> None:
        self._piles = piles
        self._current_player = current_player
        self._turn_idx = turn_idx

    # ── GameState protocol ────────────────────────────────────────────────────

    @property
    def piles(self) -> tuple[int, ...]:
        return self._piles

    @property
    def turn_idx(self) -> int:
        return self._turn_idx

    @property
    def current_player(self) -> int:
        return self._current_player

    def is_terminal(self) -> bool:
        return all(p == 0 for p in self._piles)

    def winner(self) -> int | None:
        if not self.is_terminal():
            return None
        # The player who cannot move (current_player) loses.
        # The opponent (who just moved and took the last stone) wins.
        return -self._current_player

    def legal_moves(self) -> list[int]:
        moves: list[int] = []
        for i, pile_size in enumerate(self._piles):
            for stones in range(1, pile_size + 1):
                moves.append(_encode(i, stones))
        return moves

    def apply_move(self, move: int) -> "NimState":
        if move < 0:
            # Forced pass (should never happen in Nim with stones remaining).
            return NimState(self._piles, -self._current_player, self._turn_idx + 1)
        pile_idx, stones = _decode(move)
        new_piles = list(self._piles)
        new_piles[pile_idx] -= stones
        return NimState(
            tuple(new_piles),
            -self._current_player,
            self._turn_idx + 1,
        )

    def to_serializable(self) -> dict[str, Any]:
        return {
            "game": "nim",
            "piles": list(self._piles),
            "current_player": self._current_player,
            "turn_idx": self._turn_idx,
            "nim_sum": reduce(xor, self._piles, 0),
        }

    def render_text(self, *, show_legal: bool = True) -> str:
        player_str = "Player 1" if self._current_player == 1 else "Player 2"
        you_label = player_str

        lines: list[str] = [
            "Nim — Normal play: the player who takes the last stone wins.",
            "",
        ]
        for i, size in enumerate(self._piles):
            label = _PILE_LABELS[i]
            stones_vis = "●" * size if size <= 30 else "●" * 30 + f"…(+{size-30})"
            lines.append(f"  Pile {label}: {stones_vis}  ({size} stone{'s' if size != 1 else ''})")

        lines.append("")
        lines.append(f"You are {you_label}.  Turn {self._turn_idx}.")

        if show_legal:
            legal = self.legal_moves()
            legal_strs = [self.move_to_str(m) for m in legal]
            lines.append(f"Legal moves: {', '.join(legal_strs)}")

        return "\n".join(lines)

    def move_to_str(self, move: int) -> str:
        pile_idx, stones = _decode(move)
        label = _PILE_LABELS[pile_idx]
        return f"take {stones} from {label}"

    def str_to_move(self, s: str) -> int | None:
        """Parse a move string such as 'take 2 from A' or 'A 2' or '2 from B'."""
        s = s.strip().upper()
        # Pattern 1: "take N from X" or "N from X"
        m = re.search(r"(\d+)\s+FROM\s+([A-Z])", s)
        if m:
            stones = int(m.group(1))
            pile_idx = ord(m.group(2)) - ord("A")
            return self._validate_move(pile_idx, stones)
        # Pattern 2: "from X take N" or "pile X: N"
        m = re.search(r"(?:FROM\s+|PILE\s+)([A-Z])[^0-9]*(\d+)", s)
        if m:
            pile_idx = ord(m.group(1)) - ord("A")
            stones = int(m.group(2))
            return self._validate_move(pile_idx, stones)
        # Pattern 3: standalone "X N"
        m = re.match(r"([A-Z])\s+(\d+)$", s)
        if m:
            pile_idx = ord(m.group(1)) - ord("A")
            stones = int(m.group(2))
            return self._validate_move(pile_idx, stones)
        return None

    def legal_move_regex(self) -> str:
        """Return a Python regex that matches exactly one legal Nim move tag.

        The pattern is designed for SGLang constrained decoding:

          ``[\\s\\S]*MOVE: (pile=A take=1|pile=A take=2|...|pile=C take=7)``

        The ``[\\s\\S]*`` prefix allows any reasoning prefix (including newlines)
        before the forced MOVE tag.  Every alternative is a concrete legal move,
        so the model cannot produce an out-of-range take or a non-existent pile.

        Returns a non-regex fallback string if no legal moves exist (terminal
        state), though that case should never be reached in normal play.
        """
        pile_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options: list[str] = []
        for i, pile_size in enumerate(self._piles):
            label = pile_labels[i]
            for n in range(1, pile_size + 1):
                options.append(f"pile={label} take={n}")
        if not options:
            return r"[\s\S]*MOVE: pile=A take=1"  # unreachable guard
        return r"[\s\S]*MOVE: (" + "|".join(options) + ")"

    def _validate_move(self, pile_idx: int, stones: int) -> int | None:
        if pile_idx < 0 or pile_idx >= len(self._piles):
            return None
        if stones < 1 or stones > self._piles[pile_idx]:
            return None
        return _encode(pile_idx, stones)


def initial_state(piles: tuple[int, ...] = (3, 5, 7)) -> NimState:
    """Return a fresh Nim game with the given pile sizes.

    The default (3, 5, 7) is a classic well-known configuration.
    For generalization tests use non-canonical sizes like (4, 6, 11).
    """
    return NimState(piles)
