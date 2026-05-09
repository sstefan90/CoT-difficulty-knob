"""Tests for LLMAgent against the MockClient."""

from __future__ import annotations

import pytest

from cot_knob.agents.llm_agent import LLMAgent
from cot_knob.games.reversi import initial_state
from cot_knob.llm.mock_client import MockClient
from cot_knob.memory.full_history import FullHistoryMemory


@pytest.mark.asyncio
async def test_llm_agent_picks_legal_move_b0():
    client = MockClient()
    mem = FullHistoryMemory()
    agent = LLMAgent(client, mem, budget=0)
    s = initial_state()
    tel = await agent.choose(s, seed=0)
    assert tel.chosen_move in s.legal_moves()
    assert tel.llm_pass1_text == ""
    assert tel.llm_pass1_finish == "skipped"


@pytest.mark.asyncio
async def test_llm_agent_pass1_token_count_tracks_budget():
    client = MockClient()
    mem = FullHistoryMemory()
    s = initial_state()
    for B in (16, 64, 256):
        agent = LLMAgent(client, mem, budget=B)
        tel = await agent.choose(s, seed=0)
        # MockClient emits exactly B whitespace tokens.
        assert len(tel.llm_pass1_text.split()) == B
        assert tel.llm_pass1_tokens_out == B
        assert tel.llm_pass1_finish in ("stop", "length")


@pytest.mark.asyncio
async def test_llm_agent_chooses_legal_move_b256():
    client = MockClient()
    mem = FullHistoryMemory()
    agent = LLMAgent(client, mem, budget=256)
    s = initial_state()
    tel = await agent.choose(s, seed=42)
    assert tel.chosen_move in s.legal_moves()
    assert tel.llm_pass2_choice_text == s.move_to_str(tel.chosen_move)
