"""Smoke tests for UCTAgent."""

from __future__ import annotations

import random

import pytest

from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.games.reversi import initial_state


@pytest.mark.asyncio
async def test_uct_picks_a_legal_move():
    s = initial_state()
    agent = UCTAgent(iterations=100, seed=0)
    tel = await agent.choose(s, seed=0)
    assert tel.chosen_move in s.legal_moves()
    assert tel.uct_top3, "UCT should report at least one top-3 entry"


@pytest.mark.asyncio
async def test_uct_beats_random_majority(  # slow-ish but bounded
):
    """At 200 iterations UCT should beat a uniform-random opponent over a small N."""
    wins = 0
    n = 5
    for game_seed in range(n):
        s = initial_state()
        uct = UCTAgent(iterations=200, seed=game_seed)
        rng = random.Random(game_seed + 1000)
        pass_streak = 0
        while not s.is_terminal() and pass_streak < 2:
            if s.current_player == 1:  # UCT plays Black
                tel = await uct.choose(s, seed=game_seed)
                m = tel.chosen_move
            else:
                moves = s.legal_moves()
                m = rng.choice(moves) if moves else -1
            if m < 0:
                pass_streak += 1
                s = s.apply_move(-1)
            else:
                pass_streak = 0
                s = s.apply_move(m)
        if s.winner() == 1:
            wins += 1
    assert wins >= 3, f"UCT(200) should beat random in majority of {n} games, got {wins}"


@pytest.mark.asyncio
async def test_uct_top3_format():
    s = initial_state()
    agent = UCTAgent(iterations=50, seed=0)
    tel = await agent.choose(s, seed=0)
    for entry in tel.uct_top3:
        assert {"move", "visits", "win_rate"} <= entry.keys()
        assert isinstance(entry["move"], str)
        assert entry["visits"] >= 1
        assert 0.0 <= entry["win_rate"] <= 1.0
