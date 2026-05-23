"""Naive bot implementations for non-LLM Avalon players.

Strategy matches the AvalonBench naive baseline (Light et al., NeurIPS 2023 GamesAndAI):
- Merlin:   proposes all-good teams; rejects teams with evil members.
- Servant:  proposes random teams; always approves (no info about evil).
- Evil:     always includes at least one evil on proposed team; approves teams
            with evil; fails every quest they're on.
- Assassin: guesses Merlin's identity randomly at assassination phase.

These bots provide the empirical anchor that LLM performance is compared against.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalon_llm.engine import AvalonGameEnvironment


def naive_choose_team(
    env: "AvalonGameEnvironment",
    leader: int,
    rng: np.random.Generator,
) -> frozenset:
    """Propose a quest team for ``leader`` using the naive strategy."""
    team_size = env.get_team_size()
    _role_id, role_name, is_good = env.get_role(leader)
    all_players = list(range(env.num_players))

    if role_name == "Merlin":
        good_players = [i for i in all_players if bool(env.is_good[i])]
        chosen = rng.choice(good_players, min(team_size, len(good_players)), replace=False)
        return frozenset(int(x) for x in chosen)

    if not is_good:
        evil_players = [i for i in all_players if not bool(env.is_good[i])]
        non_evil = [i for i in all_players if bool(env.is_good[i])]
        max_evil = min(len(evil_players), team_size)
        n_evil = int(rng.integers(1, max_evil + 1))
        evil_chosen = list(int(x) for x in rng.choice(evil_players, n_evil, replace=False))
        n_good = team_size - n_evil
        good_chosen = (
            [int(x) for x in rng.choice(non_evil, n_good, replace=False)]
            if n_good > 0 and non_evil
            else []
        )
        return frozenset(evil_chosen + good_chosen)

    # Loyal Servant / Percival — random team
    chosen = rng.choice(all_players, team_size, replace=False)
    return frozenset(int(x) for x in chosen)


def naive_team_vote(env: "AvalonGameEnvironment", player: int) -> int:
    """Return 1 (approve) or 0 (reject) for the current team proposal."""
    _role_id, role_name, is_good = env.get_role(player)
    current_team = list(env.get_current_quest_team())

    if role_name == "Merlin":
        has_evil = any(not bool(env.is_good[p]) for p in current_team)
        return 0 if has_evil else 1

    if not is_good:
        has_evil = any(not bool(env.is_good[p]) for p in current_team)
        return 1 if has_evil else 0

    # Servant / Percival: always approve (naive — no evil info)
    return 1


def naive_quest_vote(env: "AvalonGameEnvironment", player: int) -> int:
    """Return 1 (success) or 0 (fail) for the current quest."""
    _role_id, _role_name, is_good = env.get_role(player)
    return 1 if is_good else 0


def naive_assassination_target(
    env: "AvalonGameEnvironment",
    rng: np.random.Generator,
) -> int:
    """Assassin picks a random non-assassin player as the assassination target."""
    assassin_idx = env.get_assassin()
    candidates = [i for i in range(env.num_players) if i != assassin_idx]
    return int(rng.choice(candidates))
