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
            player_label = "B" if r.player == 1 else "W"
            score_b = sum(
                1 for row in r.state_after_serialized["board"] for v in row if v == 1
            )
            score_w = sum(
                1 for row in r.state_after_serialized["board"] for v in row if v == -1
            )
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
