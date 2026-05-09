"""Random agent — picks uniformly at random from legal moves.

Used as the zero-skill floor for the UCT strength ladder:
  ``win_rate(UCT-N vs Random)`` gives an absolute playing-strength scale
  independent of opponent calibration.

Also serves as the white-side baseline in side-balance experiments
(random vs random in Reversi has a small first-mover advantage, so
always running the LLM as Black would confound results).
"""

from __future__ import annotations

import random

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState


class RandomAgent(Agent):
    name = "random"

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed

    async def choose(self, state: GameState, *, seed: int | None = None) -> TurnTelemetry:

        legal = state.legal_moves()
        rng = random.Random(seed if seed is not None else self._seed)

        if not legal:
            chosen = -1
        else:
            chosen = rng.choice(legal)

        return TurnTelemetry(
            chosen_move=chosen,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            extra={"agent": "random"},
        )
