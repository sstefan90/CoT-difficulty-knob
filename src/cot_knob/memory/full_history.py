"""Full chronological history memory.

Renders every move ever played in a game, in order. Used as the
upper-bound condition for the memory ablation (proposal Task 5).
"""

from __future__ import annotations

from cot_knob.memory.base import MemoryManager, MemorySnapshot, TurnRecord


class FullHistoryMemory(MemoryManager):
    kind = "full_history"

    def __init__(self) -> None:
        self._records: list[TurnRecord] = []

    def reset(self) -> None:
        self._records = []

    def update(self, record: TurnRecord) -> None:
        self._records.append(record)

    async def render(self) -> MemorySnapshot:
        if not self._records:
            text = "(no moves played yet)"
        else:
            lines = []
            for r in self._records:
                tag = "X" if r.player == 1 else "O"
                lines.append(f"{r.turn_idx:>3}: {tag} {r.move_str}")
            text = "Move history:\n" + "\n".join(lines)
        return MemorySnapshot(
            text=text,
            n_chars=len(text),
            n_tokens_estimate=max(1, len(text.split())),
            kind=self.kind,
        )
