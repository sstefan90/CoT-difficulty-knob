"""Tests for MockClient."""

from __future__ import annotations

import pytest

from cot_knob.llm.factory import build_client
from cot_knob.llm.mock_client import MockClient


@pytest.mark.asyncio
async def test_mock_generate_returns_n_tokens():
    client = MockClient()
    out = await client.generate("hi", max_tokens=10)
    assert out.finish_reason in ("stop", "length")
    assert len(out.text.split()) == 10
    assert out.n_output_tokens == 10
    assert out.model == "mock-r1"


@pytest.mark.asyncio
async def test_mock_generate_zero_budget():
    client = MockClient()
    out = await client.generate("hi", max_tokens=0)
    assert out.text == ""
    assert out.n_output_tokens == 0


@pytest.mark.asyncio
async def test_mock_generate_choice_is_deterministic():
    client = MockClient()
    choices = ["a1", "b2", "c3", "d4"]
    a = await client.generate_choice("state-X", choices=choices, seed=42)
    b = await client.generate_choice("state-X", choices=choices, seed=42)
    assert a.choice_index == b.choice_index
    assert a.choice_text in choices


@pytest.mark.asyncio
async def test_mock_generate_choice_seed_changes_pick():
    client = MockClient()
    choices = ["a", "b", "c", "d", "e", "f", "g", "h"]
    picks = {
        (await client.generate_choice("p", choices=choices, seed=s)).choice_index
        for s in range(20)
    }
    # With 20 seeds and 8 choices we should see plenty of variety.
    assert len(picks) > 1


@pytest.mark.asyncio
async def test_factory_builds_mock():
    client = build_client({"backend": "mock", "name": "mock-test"})
    assert isinstance(client, MockClient)
    assert client.model == "mock-test"
