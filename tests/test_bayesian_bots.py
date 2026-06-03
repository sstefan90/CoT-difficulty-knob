"""Tests for the Bayesian Servant bot (BayesianBeliefState + helpers).

Covers:
  1. Belief initialisation and prior.
  2. Update after quest success (eliminates evil worlds).
  3. Update after quest failure (eliminates clean worlds).
  4. Full convergence over a simulated game.
  5. bayesian_choose_team: ranks players by p_evil.
  6. bayesian_team_vote: threshold-based rejection.
  7. Monte Carlo ceiling comparison: Bayesian servant bot raises ceiling vs naive.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

# Bring project source onto path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))

from cot_knob.games.avalon_bots import BayesianBeliefState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fresh_belief(observer: int = 0, n_players: int = 5, n_evil: int = 2) -> BayesianBeliefState:
    return BayesianBeliefState(observer=observer, n_players=n_players, n_evil=n_evil)


# ── 1. Prior ─────────────────────────────────────────────────────────────────

def test_prior_worlds_count():
    # C(4, 2) = 6 worlds for 5-player game, observer at P0
    b = _fresh_belief(observer=0)
    assert len(b._worlds) == 6


def test_prior_uniform():
    b = _fresh_belief(observer=0)
    weights = list(b._weights.values())
    assert all(abs(w - weights[0]) < 1e-12 for w in weights), "prior must be uniform"


def test_prior_p_evil_uniform():
    b = _fresh_belief(observer=0)
    # Each of P1–P4 should have p_evil = 2/4 = 0.5 at prior.
    for p in range(1, 5):
        assert abs(b.p_evil(p) - 0.5) < 1e-12, f"P{p} prior should be 0.5"


def test_prior_observer_p_evil_zero():
    b = _fresh_belief(observer=2)
    assert b.p_evil(2) == 0.0


def test_prior_expected_evil_size2():
    b = _fresh_belief(observer=0)
    # Team {0, 1}: observer(0)+1 player. E[evil] = 0 + 0.5 = 0.5
    assert abs(b.expected_evil_on_team(frozenset({0, 1})) - 0.5) < 1e-12
    # Team {1, 2}: E[evil] = 0.5 + 0.5 = 1.0
    assert abs(b.expected_evil_on_team(frozenset({1, 2})) - 1.0) < 1e-12


# ── 2. Update: quest success ──────────────────────────────────────────────────

def test_success_clears_evil_worlds():
    b = _fresh_belief(observer=0)
    # Quest succeeds with team {0, 1}: P1 must be Good.
    b.update(frozenset({0, 1}), succeeded=True)
    assert abs(b.p_evil(1)) < 1e-12, "P1 must be cleared after being on successful quest"


def test_success_raises_suspicion_of_absentees():
    b = _fresh_belief(observer=0)
    # Quest succeeds with team {0, 1}: P2, P3, P4 become more suspicious.
    b.update(frozenset({0, 1}), succeeded=True)
    # Before: p_evil(P2) = 0.5.  After: only worlds {(P2,P3),(P2,P4),(P3,P4)} remain.
    # p_evil(P2) = 2/3, p_evil(P3) = 2/3, p_evil(P4) = 2/3.
    for p in (2, 3, 4):
        assert abs(b.p_evil(p) - 2 / 3) < 1e-9, f"P{p} should be 2/3 after P1 cleared"


def test_success_normalised():
    b = _fresh_belief(observer=0)
    b.update(frozenset({0, 1}), succeeded=True)
    total = sum(b._weights.values())
    assert abs(total - 1.0) < 1e-12, "weights must sum to 1 after update"


def test_two_successes_identify_evil_pair():
    # If quests with {P0,P1} and {P0,P2} both succeed, then P3 and P4 must be evil.
    b = _fresh_belief(observer=0)
    b.update(frozenset({0, 1}), succeeded=True)  # clears P1
    b.update(frozenset({0, 2}), succeeded=True)  # clears P2
    assert abs(b.p_evil(1)) < 1e-9
    assert abs(b.p_evil(2)) < 1e-9
    assert abs(b.p_evil(3) - 1.0) < 1e-9, "P3 must be identified as evil"
    assert abs(b.p_evil(4) - 1.0) < 1e-9, "P4 must be identified as evil"


# ── 3. Update: quest failure ──────────────────────────────────────────────────

def test_failure_raises_suspicion_of_team():
    b = _fresh_belief(observer=0)
    # Quest fails with team {0, 1}: at least one of {0,1} is evil.  Observer=0 is Good,
    # so P1 must be evil.  This is degenerate for a 2-person team — P1 identified.
    b.update(frozenset({0, 1}), succeeded=False)
    assert abs(b.p_evil(1) - 1.0) < 1e-9, "P1 identified as evil from 2-person failed quest"


def test_failure_3person_raises_team_suspicion():
    b = _fresh_belief(observer=0)
    # Quest fails with team {0, 1, 2}: at least 1 of {1, 2} is evil (0=observer=Good).
    b.update(frozenset({0, 1, 2}), succeeded=False)
    # Worlds where neither P1 nor P2 is evil: only {(P3,P4)}.  That world is eliminated.
    # Remaining: all worlds that include P1 or P2 in the evil pair.
    assert b._weights[(3, 4)] < 1e-12 or (3, 4) not in b._weights, \
        "world (P3,P4) eliminated because quest failed with {P0,P1,P2} — no evil on team"
    # p_evil(P3) and p_evil(P4) should be lower than before.
    assert b.p_evil(3) < 0.5
    assert b.p_evil(4) < 0.5


# ── 4. Entropy decreases with evidence ───────────────────────────────────────

def test_entropy_decreases_after_updates():
    b = _fresh_belief(observer=0)
    h0 = b.entropy()
    b.update(frozenset({0, 1}), succeeded=True)
    h1 = b.entropy()
    b.update(frozenset({0, 2}), succeeded=True)
    h2 = b.entropy()
    assert h1 < h0, "entropy should decrease after first update"
    assert h2 < h1, "entropy should decrease further after second update"
    assert abs(h2) < 1e-9, "entropy = 0 after evil pair fully identified"


# ── 5. bayesian_choose_team (via mock env) ────────────────────────────────────

class _MockEnv:
    """Minimal stub for team-proposal tests."""
    def __init__(self, team_size: int, roles: dict[int, str], is_good: dict[int, bool]):
        self.num_players = len(roles)
        self._team_size = team_size
        self._roles = roles
        self._is_good = is_good
        self.is_good = [is_good[i] for i in range(self.num_players)]

    def get_team_size(self) -> int:
        return self._team_size

    def get_role(self, player: int):
        role = self._roles[player]
        return (0, role, self._is_good[player])

    def get_current_quest_team(self):
        return []


def test_bayesian_choose_team_after_evidence():
    from cot_knob.games.avalon_bots import bayesian_choose_team
    # 5 players: P0=Servant(LLM), P1=Merlin, P2=Servant(bot), P3=Minion, P4=Assassin
    env = _MockEnv(
        team_size=2,
        roles={0: "Servant", 1: "Merlin", 2: "Servant", 3: "Minion", 4: "Assassin"},
        is_good={0: True, 1: True, 2: True, 3: False, 4: False},
    )
    belief = _fresh_belief(observer=2)  # P2 is the Servant bot
    # After clear evidence: P1 is Good (quest succeeded with {1, 2}), P3 and P4 are evil.
    belief.update(frozenset({1, 2}), succeeded=True)   # P1 cleared
    belief.update(frozenset({0, 1}), succeeded=True)   # P0 cleared
    # Now P3 and P4 are fully identified as evil.

    rng = np.random.default_rng(42)
    team = bayesian_choose_team(env, leader=2, belief=belief, rng=rng)
    # Must not include P3 or P4 (known evil) since they have p_evil=1.0
    assert 3 not in team and 4 not in team, f"should avoid known evil players, got {team}"


def test_bayesian_choose_team_at_prior_size_equals_team():
    from cot_knob.games.avalon_bots import bayesian_choose_team
    env = _MockEnv(
        team_size=2,
        roles={0: "Servant", 1: "Merlin", 2: "Servant", 3: "Minion", 4: "Assassin"},
        is_good={0: True, 1: True, 2: True, 3: False, 4: False},
    )
    belief = _fresh_belief(observer=2)
    rng = np.random.default_rng(0)
    team = bayesian_choose_team(env, leader=2, belief=belief, rng=rng)
    assert len(team) == 2


# ── 6. bayesian_team_vote ─────────────────────────────────────────────────────

def test_bayesian_vote_rejects_known_evil_team():
    from cot_knob.games.avalon_bots import bayesian_team_vote

    class _MockEnvVote(_MockEnv):
        def get_current_quest_team(self):
            return [3, 4]  # both known evil

    env = _MockEnvVote(
        team_size=2,
        roles={0: "Servant", 1: "Merlin", 2: "Servant", 3: "Minion", 4: "Assassin"},
        is_good={0: True, 1: True, 2: True, 3: False, 4: False},
    )
    belief = _fresh_belief(observer=2)
    # After 2 successes: P0 and P1 cleared → P3 and P4 are evil with certainty.
    belief.update(frozenset({0, 2}), succeeded=True)
    belief.update(frozenset({1, 2}), succeeded=True)

    vote = bayesian_team_vote(env, player=2, belief=belief, threshold=1.0)
    assert vote == 0, "should reject team of two known evil players"


def test_bayesian_vote_approves_known_clean_team():
    from cot_knob.games.avalon_bots import bayesian_team_vote

    class _MockEnvVote(_MockEnv):
        def get_current_quest_team(self):
            return [0, 1]  # known good

    env = _MockEnvVote(
        team_size=2,
        roles={0: "Servant", 1: "Merlin", 2: "Servant", 3: "Minion", 4: "Assassin"},
        is_good={0: True, 1: True, 2: True, 3: False, 4: False},
    )
    belief = _fresh_belief(observer=2)
    belief.update(frozenset({0, 2}), succeeded=True)
    belief.update(frozenset({1, 2}), succeeded=True)

    vote = bayesian_team_vote(env, player=2, belief=belief, threshold=1.0)
    assert vote == 1, "should approve team of two known good players"


# ── 7. Monte Carlo ceiling: Bayesian servant > naive servant ──────────────────

def _simulate_ceiling_mc(
    n_sims: int = 200_000,
    bot_strategy: str = "naive",  # "naive" | "bayesian"
    rng_seed: int = 42,
) -> float:
    """Simulate games with a perfect LLM proposer; return good-side win rate.

    Game config: 5 players, quest sizes [2,3,2,3,3], no double-fail rule.
    Perfect LLM at P0: always proposes a clean team when leader.
    Bot roles: P1-P4 are Merlin, Servant, Minion, Assassin (shuffled each game).
    Assassination: random from 4 non-Assassin players → P(Merlin survives) = 3/4.

    Naive bots: Servant always approves; Merlin rejects evil proposals.
    Bayesian bots: Servant uses BayesianBeliefState to propose and vote.
    """
    from math import comb

    QUEST_SIZES = [2, 3, 2, 3, 3]
    N_PLAYERS = 5
    P_MERLIN_SURVIVES = 3 / 4

    rng = np.random.default_rng(rng_seed)
    wins = 0

    for _ in range(n_sims):
        # Role assignment: P0 = LLM (perfect proposer), P1-P4 shuffled.
        non_llm = rng.permutation(["Merlin", "Servant", "Minion", "Assassin"]).tolist()
        roles = ["LLM"] + non_llm
        is_good = [True if r in ("LLM", "Merlin", "Servant") else False for r in roles]
        evil_players = [i for i, g in enumerate(is_good) if not g]

        # For Bayesian strategy, each Servant-role bot gets its own belief state.
        servant_beliefs: dict[int, BayesianBeliefState] = {}
        if bot_strategy == "bayesian":
            for p in range(1, N_PLAYERS):
                if roles[p] == "Servant":
                    servant_beliefs[p] = BayesianBeliefState(
                        observer=p, n_players=N_PLAYERS, n_evil=2
                    )

        start_leader = int(rng.integers(0, N_PLAYERS))
        # Track pending rejections (Avalon: 5 consecutive = evil wins).
        consecutive_rejects = 0
        good_q = 0
        evil_q = 0
        quest_idx = 0
        current_leader = start_leader
        evil_auto_win = False

        while good_q < 3 and evil_q < 3 and quest_idx < 5:
            k = QUEST_SIZES[quest_idx]
            leader_role = roles[current_leader]

            # ── Proposal ────────────────────────────────────────────────────
            if leader_role in ("LLM", "Merlin"):
                # Perfect proposer: always picks a clean team of size k.
                good_players = [i for i, g in enumerate(is_good) if g]
                team = frozenset(rng.choice(good_players, min(k, len(good_players)), replace=False).tolist())
            elif leader_role == "Servant" and bot_strategy == "bayesian":
                bl = servant_beliefs[current_leader]
                noise = rng.random(N_PLAYERS)
                sorted_p = sorted(range(N_PLAYERS), key=lambda i: (bl.p_evil(i), noise[i]))
                team = frozenset(sorted_p[:k])
            elif leader_role == "Servant":
                # Naive: random team.
                team = frozenset(int(x) for x in rng.choice(N_PLAYERS, k, replace=False))
            else:
                # Evil: include at least one evil player.
                n_evil_on_team = int(rng.integers(1, min(len(evil_players), k) + 1))
                evil_chosen = rng.choice(evil_players, n_evil_on_team, replace=False).tolist()
                good_pool = [i for i, g in enumerate(is_good) if g]
                n_good_on_team = k - n_evil_on_team
                good_chosen = rng.choice(good_pool, n_good_on_team, replace=False).tolist() if n_good_on_team > 0 else []
                team = frozenset(evil_chosen + good_chosen)

            # ── Voting ──────────────────────────────────────────────────────
            approve_count = 0
            for voter in range(N_PLAYERS):
                role_v = roles[voter]
                team_has_evil = any(e in team for e in evil_players)
                if role_v == "Merlin":
                    vote = 0 if team_has_evil else 1
                elif role_v in ("Minion", "Assassin"):
                    vote = 1 if team_has_evil else 0
                elif role_v == "LLM":
                    # LLM votes to approve its own team (coherent at high budget).
                    vote = 1
                elif role_v == "Servant" and bot_strategy == "bayesian":
                    bl = servant_beliefs[voter]
                    vote = 0 if bl.expected_evil_on_team(team) >= 1.0 else 1
                else:
                    vote = 1  # naive Servant always approves
                approve_count += vote

            if approve_count < 3:
                # Proposal rejected — rotate leader, don't advance quest.
                consecutive_rejects += 1
                if consecutive_rejects >= 5:
                    evil_auto_win = True
                    break
                current_leader = (current_leader + 1) % N_PLAYERS
                continue

            consecutive_rejects = 0  # accepted

            # ── Quest execution ──────────────────────────────────────────────
            quest_has_evil = any(e in team for e in evil_players)
            if quest_has_evil:
                evil_q += 1
                succeeded = False
            else:
                good_q += 1
                succeeded = True

            # Bayesian update after quest result.
            if bot_strategy == "bayesian":
                for bl in servant_beliefs.values():
                    bl.update(team, succeeded)

            quest_idx += 1
            current_leader = (current_leader + 1) % N_PLAYERS

        if evil_auto_win or evil_q >= 3:
            continue
        if good_q >= 3 and rng.random() < P_MERLIN_SURVIVES:
            wins += 1

    return wins / n_sims


@pytest.mark.slow
def test_bayesian_ceiling_exceeds_naive():
    """Bayesian Servant bots should give the good side a higher ceiling than naive."""
    naive_wr = _simulate_ceiling_mc(n_sims=100_000, bot_strategy="naive", rng_seed=0)
    bayes_wr = _simulate_ceiling_mc(n_sims=100_000, bot_strategy="bayesian", rng_seed=0)
    print(f"\n  Naive ceiling:    {naive_wr:.1%}")
    print(f"  Bayesian ceiling: {bayes_wr:.1%}")
    assert bayes_wr > naive_wr, (
        f"Bayesian Servant should raise the ceiling above naive ({bayes_wr:.1%} vs {naive_wr:.1%})"
    )


def test_bayesian_ceiling_quick():
    """Smoke test: both strategies produce reasonable win rates (> 0, < 100%)."""
    for strategy in ("naive", "bayesian"):
        wr = _simulate_ceiling_mc(n_sims=5_000, bot_strategy=strategy, rng_seed=1)
        assert 0.0 < wr < 1.0, f"{strategy} strategy produced degenerate win rate {wr}"
