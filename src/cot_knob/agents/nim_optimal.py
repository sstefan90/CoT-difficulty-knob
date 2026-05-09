"""Nim optimal agent — exact Sprague-Grundy / nim-sum oracle.

For any position with nim_sum != 0 the agent plays the unique winning move.
For losing positions (nim_sum == 0) it plays a random legal move (cannot
influence the outcome under perfect play by the opponent).

This agent is used both as:
1. A game opponent (``opponent.kind: nim_optimal`` in SweepConfig).
2. An oracle for move evaluation (replacing UCT-2000 for Nim sweeps).

Oracle semantics (parallel to UCTAgent):
- ``uct_top3`` field contains all legal moves sorted by outcome value (1 =
  winning, 0 = losing).  The winning move, if it exists, is ranked first.
- ``move_quality`` = 1 if the chosen move is a winning move, else 0.
"""

from __future__ import annotations

import random
from functools import reduce
from operator import xor
from typing import Any

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState
from cot_knob.games.nim import NimState, _decode, _encode


def _nim_winning_moves(piles: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return all (pile_idx, stones_to_take) pairs that are nim-winning moves.

    Returns an empty list when nim_sum == 0 (current player is in a losing
    position under perfect play — any move is valid but all lose).
    """
    nim_sum = reduce(xor, piles, 0)
    if nim_sum == 0:
        return []
    winning: list[tuple[int, int]] = []
    for i, p in enumerate(piles):
        target = p ^ nim_sum
        if target < p:
            winning.append((i, p - target))
    return winning


class NimOptimalAgent(Agent):
    """Plays nim-optimal strategy; used as both opponent and oracle."""

    name = "nim_optimal"

    def __init__(self, *, seed: int | None = None, top_k: int | None = 3) -> None:
        self._seed = seed
        self._top_k = top_k

    async def choose(
        self,
        state: GameState,
        *,
        seed: int | None = None,
    ) -> TurnTelemetry:
        if not isinstance(state, NimState):
            raise TypeError(
                f"NimOptimalAgent supports NimState only, got {type(state)}"
            )

        legal = state.legal_moves()
        if not legal:
            return TurnTelemetry(
                chosen_move=-1,
                legal_moves=[],
                state_serialized=state.to_serializable(),
                extra={"agent": "nim_optimal", "nim_sum": 0},
            )

        piles = state.piles
        winning_list = _nim_winning_moves(piles)
        in_losing_pos = len(winning_list) == 0

        # Build set of winning move IDs for oracle ratings.
        winning_ids: set[int] = {_encode(i, s) for i, s in winning_list}

        # Oracle rankings: all winning moves get win_rate=1.0; all losing
        # moves get 0.0 (or 0.5 when already in a losing position, to signal
        # uniform uncertainty rather than false precision).
        uct_top3_raw: list[dict[str, Any]] = []
        for m in legal:
            if m in winning_ids:
                wr = 1.0
            elif in_losing_pos:
                wr = 0.5  # all moves lose; rank uniformly
            else:
                wr = 0.0
            uct_top3_raw.append({"move": state.move_to_str(m), "win_rate": wr, "move_id": m})

        uct_top3_raw.sort(key=lambda x: -x["win_rate"])

        top_k = self._top_k
        uct_top3 = uct_top3_raw if top_k is None else uct_top3_raw[:top_k]

        # Select move.
        if winning_list:
            # Pick the first winning move (deterministic and correct).
            chosen = _encode(winning_list[0][0], winning_list[0][1])
        else:
            rng = random.Random(seed if seed is not None else self._seed)
            chosen = rng.choice(legal)

        return TurnTelemetry(
            chosen_move=chosen,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            uct_top3=uct_top3,
            extra={
                "agent": "nim_optimal",
                "nim_sum": reduce(xor, piles, 0),
                "in_losing_pos": in_losing_pos,
                "n_winning_moves": len(winning_list),
            },
        )
