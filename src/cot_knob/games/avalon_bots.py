"""Bot implementations for non-LLM Avalon players.

Two strategies are provided:

Naive strategy (Light et al., NeurIPS 2023 GamesAndAI baseline):
- Merlin:   proposes all-good teams; rejects teams with evil members.
- Servant:  proposes random teams; always approves (no info about evil).
- Evil:     always includes at least one evil on proposed team; approves teams
            with evil; fails every quest they're on.
- Assassin: guesses Merlin's identity randomly at assassination phase.

Bayesian strategy (matches the Servant update from the AvalonBench paper):
- Merlin/Evil/Assassin: identical to naive.
- Servant: maintains a posterior over the C(n_players-1, n_evil) possible evil
  assignments and updates it after each quest outcome.  Proposes the team that
  minimises expected evil; votes to reject if expected evil on the proposed
  team meets or exceeds a configurable threshold (default 1.0).

These bots provide the empirical anchor that LLM performance is compared
against.  The Bayesian Servant makes votes meaningful: it blocks suspicious
proposals, removing the naive-bot vote-blocking loophole.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from avalon_llm.engine import AvalonGameEnvironment


# ── Naive strategy ──────────────────────────────────────────────────────────

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

    # Loyal Servant — random team
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

    # Servant: always approve (naive — no evil info)
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


# ── Bayesian strategy ────────────────────────────────────────────────────────

class BayesianBeliefState:
    """Posterior distribution over evil assignments for a Servant observer.

    Maintains a joint distribution over all C(n_players - 1, n_evil) possible
    evil assignments, updated by exact Bayesian inference after each quest.

    With naive bots (evil always fails quests):
      quest succeeds  →  every world where any team member is evil is eliminated
      quest fails     →  every world where NO team member is evil is eliminated

    This gives the tightest possible update from quest outcomes alone (no
    information from vote patterns is used, matching AvalonBench's Servant).

    Args:
        observer:   Index of the player whose perspective this models (always
                    known to be Good — excluded from evil candidate worlds).
        n_players:  Total players in the game (5 in standard Avalon).
        n_evil:     Number of evil players (2 in standard 5-player Avalon).
    """

    def __init__(self, observer: int, n_players: int, n_evil: int = 2) -> None:
        self.observer = observer
        self.n_players = n_players
        self.n_evil = n_evil
        other_players = [i for i in range(n_players) if i != observer]
        # Enumerate all C(n_players-1, n_evil) possible evil subsets.
        self._worlds: list[tuple[int, ...]] = list(combinations(other_players, n_evil))
        # Uniform prior (normalised).
        prior = 1.0 / len(self._worlds)
        self._weights: dict[tuple[int, ...], float] = {w: prior for w in self._worlds}

    # ── Posterior update ────────────────────────────────────────────────────

    def update(self, team: frozenset[int], succeeded: bool) -> None:
        """Bayesian update from a single quest outcome.

        Because evil bots always vote fail, quest success is a perfect signal
        that no evil player was on the team.
        """
        team_set = set(team)
        for world in self._worlds:
            evil_on_team = any(e in team_set for e in world)
            if succeeded and evil_on_team:
                self._weights[world] = 0.0
            elif not succeeded and not evil_on_team:
                self._weights[world] = 0.0
        total = sum(self._weights.values())
        if total > 0:
            for w in self._worlds:
                self._weights[w] /= total

    # ── Marginal queries ────────────────────────────────────────────────────

    def p_evil(self, player: int) -> float:
        """P(player is evil) under the current posterior."""
        if player == self.observer:
            return 0.0
        return sum(self._weights[w] for w in self._worlds if player in w)

    def expected_evil_on_team(self, team: frozenset[int]) -> float:
        """E[number of evil players on team] under the current posterior."""
        return sum(self.p_evil(p) for p in team)

    def most_likely_evil_pair(self) -> tuple[int, ...] | None:
        """Return the most probable evil assignment, or None if belief is flat."""
        if not self._worlds:
            return None
        return max(self._worlds, key=lambda w: self._weights[w])

    def entropy(self) -> float:
        """Shannon entropy of the belief (in nats). Zero = certain."""
        import math
        h = 0.0
        for w, wt in self._weights.items():
            if wt > 0:
                h -= wt * math.log(wt)
        return h


# ── Bayesian bot actions ─────────────────────────────────────────────────────

def bayesian_choose_team(
    env: "AvalonGameEnvironment",
    leader: int,
    belief: BayesianBeliefState,
    rng: np.random.Generator,
) -> frozenset:
    """Propose a quest team using the Bayesian Servant strategy.

    Merlin and evil roles use their naive strategies unchanged.
    Servant ranks all players by p_evil (ascending) and picks the k cheapest,
    breaking ties uniformly at random.
    """
    team_size = env.get_team_size()
    _role_id, role_name, is_good = env.get_role(leader)
    all_players = list(range(env.num_players))

    if role_name == "Merlin":
        good_players = [i for i in all_players if bool(env.is_good[i])]
        chosen = rng.choice(good_players, min(team_size, len(good_players)), replace=False)
        return frozenset(int(x) for x in chosen)

    if not is_good:
        return naive_choose_team(env, leader, rng)

    # Servant: pick k players with lowest expected evil contribution.
    # Random noise breaks ties so the ordering isn't systematically biased
    # toward low-index players when probabilities are equal (e.g., at prior).
    noise = rng.random(len(all_players))
    sorted_players = sorted(all_players, key=lambda i: (belief.p_evil(i), noise[i]))
    return frozenset(sorted_players[:team_size])


def bayesian_team_vote(
    env: "AvalonGameEnvironment",
    player: int,
    belief: BayesianBeliefState,
    threshold: float = 1.0,
) -> int:
    """Vote on the current team proposal using the Bayesian Servant strategy.

    Merlin and evil roles use their naive strategies unchanged.
    Servant rejects if E[evil on team] >= threshold (default 1.0, meaning:
    reject any team where you expect at least one evil member on average).

    The threshold controls how aggressive the Servant is:
      0.5  – reject any team with non-trivial suspicion (very aggressive)
      1.0  – reject teams where you expect ≥1 evil member (balanced)
      1.5  – reject only when you strongly expect ≥2 evil (permissive)

    A threshold of 1.0 is appropriate after a few quests have cleared some
    players; at pure prior (p_evil=0.5 per player), a size-2 team not
    including the observer has E[evil]=1.0 and is rejected.
    """
    _role_id, role_name, is_good = env.get_role(player)
    current_team = frozenset(env.get_current_quest_team())

    if role_name == "Merlin":
        has_evil = any(not bool(env.is_good[p]) for p in current_team)
        return 0 if has_evil else 1

    if not is_good:
        has_evil = any(not bool(env.is_good[p]) for p in current_team)
        return 1 if has_evil else 0

    # Servant: reject if expected evil on team meets the threshold.
    return 0 if belief.expected_evil_on_team(current_team) >= threshold else 1
