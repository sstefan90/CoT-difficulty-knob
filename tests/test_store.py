"""Tests for the SQLite Store + JSONL writer."""

from __future__ import annotations

import json
from pathlib import Path

from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


def test_store_init_creates_schema(tmp_path: Path):
    db = tmp_path / "results.db"
    store = Store(db)
    assert db.exists()
    cur = store._conn.cursor()
    rows = list(cur.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    names = {r["name"] for r in rows}
    assert {"runs", "trials", "turns", "model_calls", "summaries"}.issubset(names)
    store.close()


def test_full_insert_path(tmp_path: Path):
    store = Store(tmp_path / "results.db")
    run_id = store.insert_run(
        name="smoke", config_hash="abc", config_yaml="x: 1",
        backend="mock", model="mock-r1",
    )
    assert run_id.startswith("run_")

    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0,
        condition={"budget": 64, "model": "mock-r1"},
        seed=11, llm_side=1,
    )
    turn_id = store.insert_turn(
        trial_id=trial_id, turn_idx=0, phase="early", player=1,
        agent_kind="llm",
        board_state={"turn_idx": 0},
        legal_moves=["d3", "c4", "e6", "f5"],
        chosen_move="d3", chosen_move_id=18,
        uct_top3=[{"move": "d3", "visits": 100, "win_rate": 0.55}],
        move_quality=1, latency_ms_total=1234.0,
    )
    store.insert_model_call(
        trial_id=trial_id, turn_id=turn_id, role="reason",
        prompt_text="P", response_text="R", n_input_tokens=10, n_output_tokens=64,
        finish_reason="stop", temperature=0.0, seed=11, latency_ms=1100.0,
        backend="mock", model="mock-r1",
    )
    store.insert_summary(
        trial_id=trial_id, after_turn_idx=0, summary_text="empty",
        n_tokens_est=2, kind="summary",
    )
    store.finalize_trial(
        trial_id, winner="llm", n_turns_total=60, n_llm_turns=30,
        n_uct_turns=30, final_score_llm=33, final_score_uct=27,
    )
    store.finalize_run(run_id)

    cur = store._conn.cursor()
    row = cur.execute("SELECT * FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    assert row["winner"] == "llm"
    cond = json.loads(row["condition_json"])
    assert cond["budget"] == 64

    counts = {
        "turns": cur.execute("SELECT COUNT(*) AS c FROM turns").fetchone()["c"],
        "calls": cur.execute("SELECT COUNT(*) AS c FROM model_calls").fetchone()["c"],
        "summaries": cur.execute("SELECT COUNT(*) AS c FROM summaries").fetchone()["c"],
    }
    assert counts == {"turns": 1, "calls": 1, "summaries": 1}
    store.close()


def test_jsonl_writer_appends(tmp_path: Path):
    w = JSONLWriter(tmp_path / "runs", "run_xyz")
    w.write("trial_1", "turn", {"a": 1})
    w.write("trial_1", "turn", {"a": 2})
    w.close()
    p = tmp_path / "runs" / "run_xyz" / "trial_1.jsonl"
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
