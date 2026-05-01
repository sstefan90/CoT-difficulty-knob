"""Tests for memory managers."""

from __future__ import annotations

import pytest

from cot_knob.llm.mock_client import MockClient
from cot_knob.memory.base import TurnRecord
from cot_knob.memory.full_history import FullHistoryMemory
from cot_knob.memory.summary import StructuredSummaryMemory


def _rec(idx: int, mv: str) -> TurnRecord:
    return TurnRecord(
        turn_idx=idx,
        player=1 if idx % 2 == 0 else -1,
        move_str=mv,
        state_after_serialized={"turn_idx": idx},
    )


@pytest.mark.asyncio
async def test_full_history_renders_chronologically():
    m = FullHistoryMemory()
    m.update(_rec(0, "d3"))
    m.update(_rec(1, "c5"))
    snap = await m.render()
    assert "d3" in snap.text and "c5" in snap.text
    # d3 before c5
    assert snap.text.index("d3") < snap.text.index("c5")
    assert snap.kind == "full_history"


@pytest.mark.asyncio
async def test_summary_caches_and_refreshes():
    client = MockClient()
    m = StructuredSummaryMemory(client, max_summary_tokens=32, summarize_every=2)
    m.update(_rec(0, "d3"))
    s1 = await m.render()
    m.update(_rec(1, "c5"))
    s2 = await m.render()  # within `every`, should reuse
    m.update(_rec(2, "f4"))
    m.update(_rec(3, "e3"))
    s3 = await m.render()  # crossed the threshold, should refresh
    assert s1.kind == s2.kind == s3.kind == "summary"
    assert s1.text != s3.text  # the recent-moves tail differs at minimum


@pytest.mark.asyncio
async def test_summary_empty_state():
    client = MockClient()
    m = StructuredSummaryMemory(client)
    snap = await m.render()
    assert "no moves" in snap.text.lower()
