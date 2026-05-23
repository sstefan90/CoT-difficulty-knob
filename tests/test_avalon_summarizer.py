"""Tests for the AvalonSummarizer.

Two test groups:
1. Unit tests for the summarizer helper functions (no API calls).
2. Integration tests that mock the Anthropic client to verify store/JSONL writes.

Summary *effectiveness* tests (do summaries correctly identify key facts?)
require real model calls and are intentionally deferred — the full game-state
and summary are already persisted in SQLite/JSONL so they can be run offline
against stored traces at any time.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cot_knob.games.avalon_summarizer import AvalonSummarizer, _format_event_log
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


# ── Unit tests for helpers ─────────────────────────────────────────────────

def test_format_event_log_empty():
    assert _format_event_log([]) == "(no events yet)"


def test_format_event_log_vote_result():
    history = [{
        "type": "team_vote_result",
        "quest_turn": 0,
        "round": 0,
        "leader": 2,
        "team": [1, 2],
        "votes": [1, 1, 1, 0, 0],
        "accepted": True,
    }]
    out = _format_event_log(history)
    assert "Quest 1" in out
    assert "P2 proposed" in out
    assert "ACCEPTED" in out


def test_format_event_log_quest_result():
    history = [{
        "type": "quest_result",
        "quest_turn": 1,
        "team": [0, 1, 2],
        "succeeded": False,
        "num_fails": 1,
    }]
    out = _format_event_log(history)
    assert "Quest 2" in out
    assert "FAIL" in out
    assert "1 fail votes" in out


def test_format_event_log_discussion():
    history = [{
        "type": "discussion",
        "quest_turn": 0,
        "round": 0,
        "leader": 1,
        "team": [0, 1],
        "statements": {"0": "I think player 3 is suspicious."},
    }]
    out = _format_event_log(history)
    assert "discussion" in out
    assert "P0" in out


# ── Integration tests (Anthropic API mocked) ───────────────────────────────

def _make_mock_response(text: str = "Mock summary text.") -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 50
    usage.output_tokens = 20
    content_block = MagicMock()
    content_block.text = text
    resp = MagicMock()
    resp.content = [content_block]
    resp.usage = usage
    resp.stop_reason = "end_turn"
    return resp


@pytest.mark.asyncio
async def test_summarizer_stores_to_sqlite(tmp_path: Path):
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_summ")

    run_id = store.insert_run(
        name="test_summ", config_hash="s1", config_yaml="game: avalon",
        backend="mock", model="mock",
    )
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0,
        condition={"game": "avalon"}, seed=0, llm_side=0,
    )

    mock_response = _make_mock_response("Good side won quest 1. Player 3 is suspicious.")

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.messages.create = AsyncMock(return_value=mock_response)

        summarizer = AvalonSummarizer(model="claude-sonnet-4-6")
        history = [{
            "type": "quest_result",
            "quest_turn": 0,
            "team": [0, 1],
            "succeeded": True,
            "num_fails": 0,
        }]
        text = await summarizer.summarize(
            trial_id=trial_id,
            after_quest_idx=0,
            history=history,
            quest_results=[True],
            llm_player_idx=0,
            llm_role="Servant",
            store=store,
            jsonl=jsonl,
        )
        await summarizer.aclose()

    assert text == "Good side won quest 1. Player 3 is suspicious."

    cur = store._conn.cursor()
    rows = list(cur.execute("SELECT * FROM summaries WHERE trial_id=?", (trial_id,)))
    assert len(rows) == 1
    assert rows[0]["summary_text"] == text
    assert rows[0]["after_turn_idx"] == 0
    assert rows[0]["kind"] == "public"

    model_call_rows = list(
        cur.execute("SELECT * FROM model_calls WHERE role='summarize'")
    )
    assert len(model_call_rows) == 1

    jsonl.close()
    store.close()


@pytest.mark.asyncio
async def test_summarizer_writes_jsonl_event(tmp_path: Path):
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_summ2")

    run_id = store.insert_run(
        name="test_summ2", config_hash="s2", config_yaml="game: avalon",
        backend="mock", model="mock",
    )
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0,
        condition={"game": "avalon"}, seed=0, llm_side=0,
    )

    mock_response = _make_mock_response("Quest 1 succeeded with team [0,1].")

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.messages.create = AsyncMock(return_value=mock_response)

        summarizer = AvalonSummarizer(model="claude-sonnet-4-6")
        await summarizer.summarize(
            trial_id=trial_id,
            after_quest_idx=0,
            history=[],
            quest_results=[True],
            llm_player_idx=0,
            llm_role="Servant",
            store=store,
            jsonl=jsonl,
        )
        await summarizer.aclose()

    jsonl.close()
    store.close()

    jsonl_path = tmp_path / "runs" / "run_summ2" / f"{trial_id}.jsonl"
    events = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    summary_events = [e for e in events if e["event"] == "summary"]
    assert len(summary_events) == 1
    assert "summary_text" in summary_events[0]
    assert summary_events[0]["after_quest_idx"] == 0


@pytest.mark.asyncio
async def test_multiple_summaries_per_game(tmp_path: Path):
    """Summarizer should be callable once per quest (multiple times per game)."""
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_multi_summ")

    run_id = store.insert_run(
        name="test_multi", config_hash="s3", config_yaml="game: avalon",
        backend="mock", model="mock",
    )
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0,
        condition={"game": "avalon"}, seed=0, llm_side=0,
    )

    mock_response = _make_mock_response("Summary text.")

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.messages.create = AsyncMock(return_value=mock_response)

        summarizer = AvalonSummarizer(model="claude-sonnet-4-6")
        for i in range(3):
            await summarizer.summarize(
                trial_id=trial_id,
                after_quest_idx=i,
                history=[],
                quest_results=[True] * (i + 1),
                llm_player_idx=0,
                llm_role="Servant",
                store=store,
                jsonl=jsonl,
            )
        await summarizer.aclose()

    cur = store._conn.cursor()
    rows = list(cur.execute("SELECT * FROM summaries WHERE trial_id=?", (trial_id,)))
    assert len(rows) == 3
    turn_indices = sorted(r["after_turn_idx"] for r in rows)
    assert turn_indices == [0, 1, 2]

    jsonl.close()
    store.close()


# ── Deferred: summary effectiveness tests ─────────────────────────────────
#
# These tests require real Sonnet calls and access to stored traces.
# They are intentionally NOT run in CI — see docs/avalon_plan.md §P3-B.
# To run manually:
#   pytest tests/test_avalon_summarizer.py -k effectiveness -s
#
# Template for a future effectiveness test:
#
#   @pytest.mark.slow
#   async def test_summary_identifies_evil_team():
#       """Sonnet summary of a known-evil-team quest should mention suspicion
#       of the evil players by index."""
#       ...
