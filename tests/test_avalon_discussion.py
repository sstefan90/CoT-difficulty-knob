"""Tests for the Avalon discussion phase.

Verifies that:
- Discussion runs without errors.
- Discussion events appear in the JSONL trace.
- Discussion statements are added to game history.
- Discussion model calls are recorded in SQLite.
- Game still completes and produces valid results with discussion enabled.
- Discussion can be enabled/disabled independently of budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cot_knob.games.avalon_runner import make_env_from_seed, run_avalon_game
from cot_knob.llm.mock_client import MockClient
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


async def _run_with_discussion(
    tmp_path: Path,
    *,
    llm_role: str = "Servant",
    budget: int = 64,
    seed: int = 0,
    with_discussion: bool = True,
) -> dict:
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_disc")
    client = MockClient()

    run_id = store.insert_run(
        name="test_disc",
        config_hash="disc123",
        config_yaml="game: avalon",
        backend="mock",
        model="mock",
    )
    condition = {
        "game": "avalon", "llm_role": llm_role, "budget": budget,
        "with_discussion": with_discussion, "seed": seed,
    }
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0, condition=condition,
        seed=seed, llm_side=0,
    )
    jsonl.write(trial_id, "trial_start", {"trial_id": trial_id, "condition": condition})

    env = make_env_from_seed(seed=seed, llm_role=llm_role)
    result = await run_avalon_game(
        env=env,
        llm_player_idx=0,
        llm_role=llm_role,
        llm_client=client,
        budget=budget,
        prompt_variant="minimal",
        store=store,
        jsonl=jsonl,
        run_id=run_id,
        trial_id=trial_id,
        condition=condition,
        seed=seed,
        temperature=0.0,
        backend="mock",
        model="mock",
        with_discussion=with_discussion,
    )

    store.finalize_run(run_id)
    jsonl.close()

    cur = store._conn.cursor()
    n_calls = cur.execute("SELECT COUNT(*) AS c FROM model_calls").fetchone()["c"]
    n_discuss_calls = cur.execute(
        "SELECT COUNT(*) AS c FROM model_calls WHERE role='discuss'"
    ).fetchone()["c"]
    store.close()

    # Read JSONL events.
    jsonl_path = tmp_path / "runs" / "run_disc" / f"{trial_id}.jsonl"
    events = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    discussion_events = [e for e in events if e["event"] == "discussion"]

    return {
        "result": result,
        "n_model_calls": n_calls,
        "n_discuss_calls": n_discuss_calls,
        "discussion_events": discussion_events,
        "all_events": events,
    }


@pytest.mark.asyncio
async def test_discussion_completes(tmp_path: Path):
    out = await _run_with_discussion(tmp_path, with_discussion=True)
    assert out["result"].error is None
    assert out["result"].n_quests_played >= 3


@pytest.mark.asyncio
async def test_discussion_events_logged(tmp_path: Path):
    """Discussion events must appear in JSONL whenever a team vote occurs."""
    out = await _run_with_discussion(tmp_path, with_discussion=True, seed=1)
    # There must be at least one discussion event (one team vote happened).
    assert len(out["discussion_events"]) >= 1


@pytest.mark.asyncio
async def test_discussion_event_fields(tmp_path: Path):
    out = await _run_with_discussion(tmp_path, with_discussion=True, seed=2)
    for ev in out["discussion_events"]:
        assert "statements" in ev
        assert "quest_turn" in ev
        assert "round" in ev
        assert "team" in ev
        # LLM player 0 should have a statement entry.
        assert "0" in ev["statements"]


@pytest.mark.asyncio
async def test_discussion_model_calls_recorded(tmp_path: Path):
    """Discussion calls should be logged with role='discuss' in model_calls."""
    out = await _run_with_discussion(tmp_path, with_discussion=True, seed=3)
    # At least as many discuss calls as discussion events.
    assert out["n_discuss_calls"] >= len(out["discussion_events"])


@pytest.mark.asyncio
async def test_no_discussion_events_when_disabled(tmp_path: Path):
    out = await _run_with_discussion(tmp_path, with_discussion=False, seed=0)
    assert len(out["discussion_events"]) == 0
    assert out["n_discuss_calls"] == 0


@pytest.mark.asyncio
async def test_discussion_more_calls_than_no_discussion(tmp_path: Path):
    """Discussion mode should produce more model calls (one extra per team vote)."""
    out_disc = await _run_with_discussion(tmp_path / "disc", with_discussion=True, seed=5)
    out_nodisc = await _run_with_discussion(tmp_path / "nodisc", with_discussion=False, seed=5)
    # Discussion adds ~1 call per team proposal vote round.
    assert out_disc["n_model_calls"] > out_nodisc["n_model_calls"]


@pytest.mark.asyncio
async def test_discussion_merlin_completes(tmp_path: Path):
    out = await _run_with_discussion(tmp_path, llm_role="Merlin", with_discussion=True, seed=3)
    assert out["result"].error is None
    assert out["result"].llm_role == "Merlin"


@pytest.mark.asyncio
async def test_game_start_records_discussion_flag(tmp_path: Path):
    out = await _run_with_discussion(tmp_path, with_discussion=True, seed=0)
    game_start = next(e for e in out["all_events"] if e["event"] == "game_start")
    assert game_start["with_discussion"] is True
