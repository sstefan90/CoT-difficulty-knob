"""Last-move memory — one-liner context for perfect-information games.

Reversi is a perfect-information Markov game: the board state is the
complete sufficient statistic for optimal play. The ASCII board already
appears in every prompt, so the memory field only needs to anchor the
model in the game flow (turn number, score, whose move it was).

This memory manager emits a single line per turn, e.g.::

    Turn 5. Score: B=6 W=4. Last move: W f6.

That is intentionally minimal — the board shown in the prompt carries all
the strategic information. Contrast with ``full_history``, which gives
the full transcript and lets the model track positional patterns at the
cost of extra tokens.
"""

from __future__ import annotations

from cot_knob.memory.base import MemoryManager, MemorySnapshot, TurnRecord


class LastMoveMemory(MemoryManager):
    kind = "last_move"

    def __init__(self) -> None:
        self._last: TurnRecord | None = None

    def reset(self) -> None:
        self._last = None

    def update(self, record: TurnRecord) -> None:
        self._last = record

    async def render(self) -> MemorySnapshot:
        if self._last is None:
            text = "(game just started — no moves yet)"
        else:
            r = self._last
            s = r.state_after_serialized
            game = s.get("game", "reversi")

            if game == "nim":
                # Nim: show remaining pile sizes after the move.
                piles = s.get("piles", [])
                pile_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                pile_str = " ".join(
                    f"{pile_labels[i]}={p}" for i, p in enumerate(piles)
                )
                player_label = "Player 1" if r.player == 1 else "Player 2"
                text = (
                    f"Turn {r.turn_idx}. "
                    f"Piles after: {pile_str}. "
                    f"Last move: {player_label} {r.move_str}."
                )
            else:
                # Reversi: count pieces on the board.
                board = s["board"]
                score_b = sum(1 for row in board for v in row if v == 1)
                score_w = sum(1 for row in board for v in row if v == -1)
                player_label = "B" if r.player == 1 else "W"
                text = (
                    f"Turn {r.turn_idx}. "
                    f"Score: B={score_b} W={score_w}. "
                    f"Last move: {player_label} {r.move_str}."
                )
        return MemorySnapshot(
            text=text,
            n_chars=len(text),
            n_tokens_estimate=max(1, len(text.split())),
            kind=self.kind,
        )
