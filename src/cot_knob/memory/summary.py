"""Structured-summary memory (the proposal's default).

We re-summarize every ``summarize_every`` turns using the same LLM under
test. This is the *self-confound* the proposal acknowledges; Task 5
includes a stability check (BLEU + embedding similarity across budget
conditions) that this module's outputs feed directly into.

To keep the smoke sweep cheap, the summary call uses a small fixed
budget (``max_summary_tokens``) and a tightly templated prompt so it
behaves consistently across different agent budgets B.
"""

from __future__ import annotations

from cot_knob.llm.client import LLMClient
from cot_knob.memory.base import MemoryManager, MemorySnapshot, TurnRecord

SUMMARY_SYSTEM = (
    "You are a Reversi note-taker. Given a move history, produce a concise summary "
    "that captures: (a) corner / X-square activity, (b) edge control, (c) frontier and "
    "mobility trends, (d) any major captures. Be brief and factual. No advice."
)
SUMMARY_TEMPLATE = (
    "Move history so far ({n_moves} moves, last is most recent):\n"
    "{moves}\n\n"
    "Write the structured summary now (<= {budget} tokens)."
)


class StructuredSummaryMemory(MemoryManager):
    kind = "summary"

    def __init__(
        self,
        client: LLMClient,
        *,
        max_summary_tokens: int = 256,
        summarize_every: int = 4,
        seed: int | None = None,
    ) -> None:
        self._client = client
        self._max_tokens = int(max_summary_tokens)
        self._every = max(1, int(summarize_every))
        self._seed = seed

        self._records: list[TurnRecord] = []
        self._cached_summary: str | None = None
        self._cached_at_turn: int = -1

    def reset(self) -> None:
        self._records = []
        self._cached_summary = None
        self._cached_at_turn = -1

    def update(self, record: TurnRecord) -> None:
        self._records.append(record)

    def _format_moves(self) -> str:
        return ", ".join(
            f"{('X' if r.player == 1 else 'O')}:{r.move_str}" for r in self._records
        )

    async def render(self) -> MemorySnapshot:
        if not self._records:
            return MemorySnapshot(
                text="(no moves played yet)",
                n_chars=24,
                n_tokens_estimate=4,
                kind=self.kind,
            )

        # Re-summarize only every N turns to keep the smoke sweep cheap.
        n = len(self._records)
        if self._cached_summary is None or (n - self._cached_at_turn) >= self._every:
            prompt = SUMMARY_TEMPLATE.format(
                n_moves=n,
                moves=self._format_moves(),
                budget=self._max_tokens,
            )
            comp = await self._client.generate(
                prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,
                seed=self._seed,
                system=SUMMARY_SYSTEM,
            )
            self._cached_summary = comp.text.strip() or "(empty summary)"
            self._cached_at_turn = n

        text = (
            f"Structured summary (after {self._cached_at_turn} moves):\n"
            f"{self._cached_summary}\n"
            f"Most-recent moves: {', '.join(r.move_str for r in self._records[-6:])}"
        )
        return MemorySnapshot(
            text=text,
            n_chars=len(text),
            n_tokens_estimate=max(1, len(text.split())),
            kind=self.kind,
            extra={
                "n_records": n,
                "summary_token_budget": self._max_tokens,
                "summarize_every": self._every,
            },
        )
