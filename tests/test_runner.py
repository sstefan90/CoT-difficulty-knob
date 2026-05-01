"""End-to-end tests of the runner with MockClient."""

from __future__ import annotations

from pathlib import Path

import pytest

from cot_knob.agents.llm_agent import LLMAgent
from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.experiments.runner import play_match
from cot_knob.llm.mock_client import MockClient
from cot_knob.memory.full_history import FullHistoryMemory
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


@pytest.mark.asyncio
async def test_mock_full_game_with_uct(tmp_path: Path):
    client = MockClient()
    memory = FullHistoryMemory()
    llm = LLMAgent(client, memory, budget=32, side=1)
    uct = UCTAgent(iterations=50, seed=0)
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_x")
    run_id = store.insert_run(
        name="t", config_hash="h", config_yaml="x: 1",
        backend="mock", model="mock-r1",
    )
    res = await play_match(
        store=store, jsonl=jsonl, run_id=run_id, cell_index=0,
        condition={"budget": 32, "model": "mock-r1", "backend": "mock", "opponent": "uct-50",
                   "memory": "full_history", "seed": 0, "llm_side": "black"},
        seed=0, llm_side=1,
        llm_agent=llm, uct_agent=uct, max_turns_safety=200,
    )
    store.finalize_run(run_id)
    jsonl.close()
    assert res.winner in ("llm", "uct", "draw")
    assert res.n_turns >= 4
    cur = store._conn.cursor()
    n_turns = cur.execute("SELECT COUNT(*) AS c FROM turns").fetchone()["c"]
    n_calls = cur.execute("SELECT COUNT(*) AS c FROM model_calls").fetchone()["c"]
    assert n_turns == res.n_turns
    # Each LLM turn produces 2 calls (reason + select).
    assert n_calls >= 2 * res.n_llm_turns
    store.close()
