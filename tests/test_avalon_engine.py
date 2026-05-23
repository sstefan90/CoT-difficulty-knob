"""Unit tests for the vendored Avalon engine and naive bots.

These tests are deterministic (no LLM calls) and cover:
- Engine initialization via from_presets()
- Role/alignment queries
- Naive bot decisions at each phase
- Full game simulation with all-naive players
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from avalon_llm.engine import AvalonBasicConfig, AvalonGameEnvironment
from avalon_llm.avalon_exception import AvalonEnvException

from cot_knob.games.avalon_bots import (
    naive_assassination_target,
    naive_choose_team,
    naive_quest_vote,
    naive_team_vote,
)
from cot_knob.games.avalon_runner import make_env_from_seed


# ── Fixtures ───────────────────────────────────────────────────────────────

STANDARD_PRESET_SERVANT = {
    "num_players": 5,
    "quest_leader": 0,
    "role_names": ["Servant", "Merlin", "Servant", "Minion", "Assassin"],
}

STANDARD_PRESET_MERLIN = {
    "num_players": 5,
    "quest_leader": 0,
    "role_names": ["Merlin", "Servant", "Servant", "Minion", "Assassin"],
}


def make_env(preset: dict) -> AvalonGameEnvironment:
    return AvalonGameEnvironment.from_presets(preset)


# ── Config tests ───────────────────────────────────────────────────────────

def test_config_from_num_players():
    cfg = AvalonBasicConfig.from_num_players(5)
    assert cfg.num_players == 5
    assert cfg.num_good == 3
    assert cfg.num_evil == 2
    assert cfg.num_players_for_quest == [2, 3, 2, 3, 3]
    assert cfg.num_fails_for_quest == [1, 1, 1, 1, 1]


def test_config_roles_constants():
    assert AvalonBasicConfig.ROLES[0] == "Merlin"
    assert AvalonBasicConfig.ROLES[5] == "Servant"
    assert AvalonBasicConfig.ROLES[7] == "Assassin"
    assert AvalonBasicConfig.ROLES_REVERSE["Merlin"] == 0
    assert AvalonBasicConfig.ROLES_REVERSE["Assassin"] == 7


# ── Environment initialization tests ──────────────────────────────────────

def test_from_presets_role_assignments():
    env = make_env(STANDARD_PRESET_SERVANT)
    assert env.role_names == ["Servant", "Merlin", "Servant", "Minion", "Assassin"]
    assert bool(env.is_good[0])   # Servant is good
    assert bool(env.is_good[1])   # Merlin is good
    assert bool(env.is_good[2])   # Servant is good
    assert not bool(env.is_good[3])  # Minion is evil
    assert not bool(env.is_good[4])  # Assassin is evil


def test_from_presets_initial_state():
    env = make_env(STANDARD_PRESET_SERVANT)
    assert env.phase == 0
    assert env.round == 0
    assert env.turn == 0
    assert env.done is False
    assert env.quest_results == []
    assert env.get_quest_leader() == 0


def test_get_role():
    env = make_env(STANDARD_PRESET_MERLIN)
    role_id, role_name, is_good = env.get_role(0)
    assert role_name == "Merlin"
    assert is_good is True
    role_id, role_name, is_good = env.get_role(3)
    assert role_name == "Minion"
    assert is_good is False


def test_get_partial_sides_merlin():
    env = make_env(STANDARD_PRESET_MERLIN)
    sides = env.get_partial_sides(0)  # Merlin sees all
    assert list(sides) == [True, True, True, False, False]


def test_get_partial_sides_servant():
    env = make_env(STANDARD_PRESET_SERVANT)
    sides = env.get_partial_sides(0)  # Servant sees only self
    assert sides[0] is True  # self
    assert sides[1] == -1    # unknown
    assert sides[3] == -1    # unknown


def test_get_assassin():
    env = make_env(STANDARD_PRESET_SERVANT)
    assert env.get_assassin() == 4  # Assassin is player 4


# ── Engine action tests ────────────────────────────────────────────────────

def test_choose_quest_team_valid():
    env = make_env(STANDARD_PRESET_SERVANT)
    phase, done, next_leader = env.choose_quest_team(frozenset({0, 1}), leader=0)
    assert phase == 1   # moved to Team Voting
    assert done is False
    assert env.get_current_quest_team() == frozenset({0, 1})


def test_choose_quest_team_wrong_leader():
    env = make_env(STANDARD_PRESET_SERVANT)
    with pytest.raises(AvalonEnvException, match="Not quest leader"):
        env.choose_quest_team(frozenset({0, 1}), leader=1)


def test_choose_quest_team_wrong_size():
    env = make_env(STANDARD_PRESET_SERVANT)
    with pytest.raises(AvalonEnvException, match="Invalid team size"):
        env.choose_quest_team(frozenset({0, 1, 2}), leader=0)  # quest 0 needs 2 players


def test_gather_team_votes_accept():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 1}), leader=0)
    phase, done, accepted = env.gather_team_votes([1, 1, 1, 0, 0])  # 3-2 majority
    assert accepted is True
    assert phase == 2


def test_gather_team_votes_reject():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 1}), leader=0)
    phase, done, accepted = env.gather_team_votes([0, 0, 0, 1, 1])  # 2-3 rejected
    assert accepted is False
    assert phase == 0
    assert env.round == 1


def test_gather_quest_votes_success():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 2}), leader=0)
    env.gather_team_votes([1, 1, 1, 0, 0])
    phase, done, succeeded, num_fails = env.gather_quest_votes([1, 1])  # both success
    assert succeeded is True
    assert num_fails == 0
    assert env.quest_results == [True]


def test_gather_quest_votes_fail():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 3}), leader=0)  # includes evil player 3
    env.gather_team_votes([1, 1, 1, 1, 0])
    phase, done, succeeded, num_fails = env.gather_quest_votes([1, 0])  # one fail
    assert succeeded is False
    assert num_fails == 1
    assert env.quest_results == [False]


# ── Naive bot tests ────────────────────────────────────────────────────────

def test_naive_choose_team_merlin():
    env = make_env(STANDARD_PRESET_MERLIN)  # Player 0 is Merlin
    rng = np.random.default_rng(42)
    team = naive_choose_team(env, leader=0, rng=rng)
    # Merlin should only pick good players (0, 1, 2)
    assert len(team) == env.get_team_size()  # turn=0, need 2 players
    assert all(bool(env.is_good[p]) for p in team), f"Merlin picked evil: {team}"


def test_naive_choose_team_evil():
    env = make_env(STANDARD_PRESET_SERVANT)  # Player 3=Minion, 4=Assassin
    rng = np.random.default_rng(42)
    # Run multiple times to verify evil always included
    for trial_seed in range(20):
        rng = np.random.default_rng(trial_seed)
        team = naive_choose_team(env, leader=3, rng=rng)
        has_evil = any(not bool(env.is_good[p]) for p in team)
        assert has_evil, f"Evil leader didn't include evil in team: {team}"


def test_naive_team_vote_merlin_rejects_evil():
    env = make_env(STANDARD_PRESET_MERLIN)
    env.choose_quest_team(frozenset({0, 3}), leader=0)  # team has evil player 3
    vote = naive_team_vote(env, player=0)  # Merlin should reject
    assert vote == 0


def test_naive_team_vote_merlin_approves_good():
    env = make_env(STANDARD_PRESET_MERLIN)
    env.choose_quest_team(frozenset({0, 1}), leader=0)  # all-good team
    vote = naive_team_vote(env, player=0)  # Merlin should approve
    assert vote == 1


def test_naive_team_vote_servant_always_approves():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 3}), leader=0)  # evil on team
    # Servant doesn't know — always approves
    vote = naive_team_vote(env, player=0)
    assert vote == 1


def test_naive_team_vote_evil_approves_evil_team():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 3}), leader=0)  # player 3 is Minion
    vote = naive_team_vote(env, player=3)  # Minion approves (team has evil)
    assert vote == 1


def test_naive_team_vote_evil_rejects_good_team():
    env = make_env(STANDARD_PRESET_SERVANT)
    env.choose_quest_team(frozenset({0, 1}), leader=0)  # all good
    vote = naive_team_vote(env, player=3)  # Minion rejects
    assert vote == 0


def test_naive_quest_vote_good_success():
    env = make_env(STANDARD_PRESET_SERVANT)
    assert naive_quest_vote(env, player=0) == 1  # Servant votes success
    assert naive_quest_vote(env, player=1) == 1  # Merlin votes success


def test_naive_quest_vote_evil_fail():
    env = make_env(STANDARD_PRESET_SERVANT)
    assert naive_quest_vote(env, player=3) == 0  # Minion votes fail
    assert naive_quest_vote(env, player=4) == 0  # Assassin votes fail


def test_naive_assassination_target():
    env = make_env(STANDARD_PRESET_SERVANT)
    rng = np.random.default_rng(0)
    target = naive_assassination_target(env, rng)
    assert target != env.get_assassin()  # can't target self
    assert 0 <= target < env.num_players


# ── make_env_from_seed tests ───────────────────────────────────────────────

def test_make_env_from_seed_servant():
    env = make_env_from_seed(seed=0, llm_role="Servant")
    role_id, role_name, is_good = env.get_role(0)
    assert role_name == "Servant"
    assert is_good is True


def test_make_env_from_seed_merlin():
    env = make_env_from_seed(seed=0, llm_role="Merlin")
    role_id, role_name, is_good = env.get_role(0)
    assert role_name == "Merlin"
    assert is_good is True


def test_make_env_from_seed_always_5_players():
    env = make_env_from_seed(seed=42, llm_role="Servant")
    assert env.num_players == 5
    good = sum(bool(env.is_good[i]) for i in range(5))
    evil = sum(not bool(env.is_good[i]) for i in range(5))
    assert good == 3
    assert evil == 2


def test_make_env_from_seed_deterministic():
    env1 = make_env_from_seed(seed=7, llm_role="Servant")
    env2 = make_env_from_seed(seed=7, llm_role="Servant")
    assert env1.role_names == env2.role_names
    assert env1.get_quest_leader() == env2.get_quest_leader()


def test_make_env_from_seed_different_seeds_differ():
    # Verify that at least some pair of seeds produce different arrangements.
    # (All-same would indicate the RNG is broken.)
    arrangements = set()
    for s in range(20):
        env = make_env_from_seed(seed=s, llm_role="Servant")
        arrangements.add((tuple(env.role_names), env.get_quest_leader()))
    assert len(arrangements) > 1, "All seeds produced identical arrangements"


# ── Full naive-vs-naive game simulation ───────────────────────────────────

def _play_naive_game(seed: int) -> dict:
    """Play a complete game with all naive bots. Returns outcome dict."""
    env = AvalonGameEnvironment.from_presets({
        "num_players": 5,
        "quest_leader": seed % 5,
        "role_names": ["Servant", "Merlin", "Servant", "Minion", "Assassin"],
    })
    rng = np.random.default_rng(seed)

    max_steps = 200
    steps = 0
    while not env.done and steps < max_steps:
        phase_id, _ = env.get_phase()
        steps += 1

        if phase_id == 0:
            leader = env.get_quest_leader()
            team = naive_choose_team(env, leader, rng)
            env.choose_quest_team(team, leader)

        elif phase_id == 1:
            votes = [naive_team_vote(env, p) for p in range(env.num_players)]
            env.gather_team_votes(votes)

        elif phase_id == 2:
            team_members = sorted(env.get_current_quest_team())
            votes = [naive_quest_vote(env, p) for p in team_members]
            env.gather_quest_votes(votes)

        elif phase_id == 3:
            assassin = env.get_assassin()
            target = naive_assassination_target(env, rng)
            env.choose_assassination_target(assassin, target)

    return {
        "good_victory": env.good_victory,
        "quest_results": list(env.quest_results),
        "done": env.done,
        "steps": steps,
    }


def test_naive_game_completes():
    """Naive-vs-naive game must terminate within 200 steps."""
    result = _play_naive_game(seed=0)
    assert result["done"] is True
    assert result["steps"] < 200


def test_naive_game_valid_outcome():
    """Outcome must be exactly 3 wins for one side."""
    result = _play_naive_game(seed=0)
    good_wins = sum(result["quest_results"])
    evil_wins = len(result["quest_results"]) - good_wins
    assert good_wins == 3 or evil_wins == 3


@pytest.mark.parametrize("seed", range(10))
def test_naive_game_completes_multiple_seeds(seed: int):
    result = _play_naive_game(seed=seed)
    assert result["done"] is True
    assert result["steps"] < 200
