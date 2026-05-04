"""UCT (UCB1-applied-to-trees) Monte-Carlo Tree Search agent for Reversi.

This is a self-contained Python UCT used for the smoke sweep on Mac. It
exposes the same ``Agent`` interface as the LLM agent, so the runner is
agnostic to which side is which.

The proposal calls for "UCT-500 / UCT-2000 / UCT-10000" baselines via
Ludii. Once Ludii.jar is available we will add ``LudiiUCTAgent`` that
delegates to Ludii's built-in UCT — same interface. Until then, this
Python UCT serves as an in-distribution stand-in: at 2000 iterations it
plays a respectable Reversi (~beats random ~95% of the time) and is
deterministic given a seed, which is what the harness needs.

The agent also exposes ``top_k`` per-move statistics so the runner can
log ``uct_top3`` for the move-quality analysis (proposal Tasks 3, 8, 12).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState
from cot_knob.games.reversi import ReversiState


@dataclass
class _Node:
    state: ReversiState
    parent: "_Node | None" = None
    move_in: int | None = None
    children: dict[int, "_Node"] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
    player_just_moved: int = 0  # the player who moved to *reach* this state

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves

    def best_child(self, c_puct: float) -> "_Node":
        log_n = math.log(self.visits) if self.visits > 0 else 0.0
        best, best_score = None, -math.inf
        for child in self.children.values():
            if child.visits == 0:
                score = math.inf
            else:
                exploit = child.wins / child.visits
                explore = c_puct * math.sqrt(log_n / child.visits)
                score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        assert best is not None
        return best


def _rollout(state: ReversiState, rng: random.Random) -> int:
    """Random playout. Returns +1 / -1 / 0 from the perspective of the original first mover."""
    s = state
    pass_streak = 0
    while not s.is_terminal() and pass_streak < 2:
        moves = s.legal_moves()
        if not moves:
            s = s.apply_move(-1)  # forced pass
            pass_streak += 1
            continue
        pass_streak = 0
        s = s.apply_move(rng.choice(moves))
    w = s.winner()
    return w if w is not None else 0


class UCTAgent(Agent):
    name = "uct"

    def __init__(
        self,
        *,
        iterations: int = 2000,
        c_puct: float = 1.41421356,
        seed: int | None = None,
        rollout_depth_cap: int | None = None,  # None = play to terminal
        top_k: int | None = 3,
    ) -> None:
        self.iterations = int(iterations)
        self.c_puct = float(c_puct)
        self._seed = seed
        self._rollout_depth_cap = rollout_depth_cap
        # top_k controls how many moves are returned in uct_top3.
        # Pass top_k=None to get ALL legal moves ranked — used by the oracle
        # in the runner to compute continuous move regret.
        self._top_k = top_k

    async def choose(self, state: GameState, *, seed: int | None = None) -> TurnTelemetry:
        if not isinstance(state, ReversiState):
            raise TypeError(f"UCTAgent currently supports ReversiState only, got {type(state)}")

        legal = state.legal_moves()
        if not legal:
            return TurnTelemetry(
                chosen_move=-1,
                legal_moves=[],
                state_serialized=state.to_serializable(),
                uct_top3=[],
                extra={"agent": "uct", "passed": True},
            )

        rng_seed = seed if seed is not None else self._seed
        rng = random.Random(rng_seed)

        root = _Node(state=state, untried_moves=list(legal))
        root_player = state.current_player

        for _ in range(self.iterations):
            node = root
            s = state

            # 1. Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_puct)
                s = s.apply_move(node.move_in)  # type: ignore[arg-type]

            # 2. Expansion (skip if terminal)
            if not s.is_terminal() and node.untried_moves:
                m = rng.choice(node.untried_moves)
                node.untried_moves.remove(m)
                child_state = s.apply_move(m)
                child = _Node(
                    state=child_state,
                    parent=node,
                    move_in=m,
                    untried_moves=child_state.legal_moves() or ([-1] if not child_state.is_terminal() else []),
                    player_just_moved=s.current_player,
                )
                node.children[m] = child
                node = child
                s = child_state

            # 3. Simulation
            result = _rollout(s, rng)

            # 4. Backpropagation. ``result`` is from Black's perspective (+1).
            # A node's win is credited when ``player_just_moved`` won.
            n: _Node | None = node
            while n is not None:
                n.visits += 1
                if n.player_just_moved != 0:
                    if result == n.player_just_moved:
                        n.wins += 1.0
                    elif result == 0:
                        n.wins += 0.5
                n = n.parent

        # Pick the most-visited child of root.
        if not root.children:
            chosen = legal[0]
        else:
            best_move, _ = max(root.children.items(), key=lambda kv: kv[1].visits)
            chosen = best_move

        # Rank all children by visits; slice to top_k (None = all).
        ranked = sorted(root.children.items(), key=lambda kv: -kv[1].visits)
        cutoff = self._top_k if self._top_k is not None else len(ranked)
        top3: list[dict] = []
        for mv, child in ranked[:cutoff]:
            wr = (child.wins / child.visits) if child.visits else 0.0
            top3.append({"move": state.move_to_str(mv), "visits": child.visits, "win_rate": wr})

        return TurnTelemetry(
            chosen_move=chosen,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            uct_top3=top3,
            extra={
                "agent": "uct",
                "iterations": self.iterations,
                "root_player": root_player,
                "n_root_children": len(root.children),
            },
        )
