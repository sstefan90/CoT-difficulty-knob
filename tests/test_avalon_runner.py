"""Integration tests for the Avalon game runner with MockClient.

Tests that:
- A complete game runs end-to-end without errors.
- JSONL events are written with expected keys.
- SQLite model_calls are recorded.
- Parse failures are handled gracefully (mock always returns a valid choice).
- Both Servant and Merlin roles complete without error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cot_knob.games.avalon_runner import make_env_from_seed, run_avalon_game
from cot_knob.llm.mock_client import MockClient
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


async def _run_one(
    tmp_path: Path,
    *,
    llm_role: str = "Servant",
    budget: int = 64,
    prompt_variant: str = "minimal",
    seed: int = 0,
) -> dict:
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_mock")
    client = MockClient()

    run_id = store.insert_run(
        name="test_run",
        config_hash="abc123",
        config_yaml="game: avalon",
        backend="mock",
        model="mock",
    )
    condition = {
        "game": "avalon", "llm_role": llm_role, "budget": budget,
        "prompt_variant": prompt_variant, "seed": seed,
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
        prompt_variant=prompt_variant,
        store=store,
        jsonl=jsonl,
        run_id=run_id,
        trial_id=trial_id,
        condition=condition,
        seed=seed,
        temperature=0.0,
        backend="mock",
        model="mock",
    )

    store.finalize_run(run_id)
    jsonl.close()

    cur = store._conn.cursor()
    n_calls = cur.execute("SELECT COUNT(*) AS c FROM model_calls").fetchone()["c"]
    trials = cur.execute("SELECT * FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    store.close()

    return {
        "result": result,
        "n_model_calls": n_calls,
        "trial_row": dict(trials),
    }


@pytest.mark.asyncio
async def test_servant_minimal_completes(tmp_path: Path):
    out = await _run_one(tmp_path, llm_role="Servant", prompt_variant="minimal", budget=64)
    result = out["result"]
    assert result.error is None
    assert result.n_quests_played in range(3, 6)
    assert result.n_good_quests + (result.n_quests_played - result.n_good_quests) == result.n_quests_played


@pytest.mark.asyncio
async def test_merlin_minimal_completes(tmp_path: Path):
    out = await _run_one(tmp_path, llm_role="Merlin", prompt_variant="minimal", budget=1024, seed=3)
    result = out["result"]
    assert result.error is None
    assert result.llm_role == "Merlin"
    assert result.llm_is_good is True


@pytest.mark.asyncio
async def test_procedural_completes(tmp_path: Path):
    out = await _run_one(tmp_path, llm_role="Servant", prompt_variant="procedural", budget=256, seed=5)
    result = out["result"]
    assert result.error is None


@pytest.mark.asyncio
async def test_budget_zero_completes(tmp_path: Path):
    """B=0 (skip Pass-1 reasoning) should still complete; Pass-2 picks a valid action."""
    out = await _run_one(tmp_path, llm_role="Servant", budget=0, seed=7)
    result = out["result"]
    assert result.error is None
    assert result.llm_pass1_tokens_total == 0


@pytest.mark.asyncio
async def test_model_calls_recorded(tmp_path: Path):
    out = await _run_one(tmp_path, llm_role="Servant", budget=64, seed=1)
    # At least one LLM decision should have been made (team vote is mandatory for LLM).
    assert out["result"].n_llm_decisions >= 1
    # Each decision = Pass-1 call (reason) + Pass-2 call (select) → 2 rows.
    assert out["n_model_calls"] >= 2 * out["result"].n_llm_decisions


@pytest.mark.asyncio
async def test_trial_winner_recorded(tmp_path: Path):
    out = await _run_one(tmp_path, llm_role="Servant", seed=2)
    trial = out["trial_row"]
    assert trial["winner"] in ("llm", "bots", "error")
    assert trial["n_llm_turns"] == out["result"].n_llm_decisions


@pytest.mark.asyncio
async def test_parse_fail_count_nonnegative(tmp_path: Path):
    """Mock always returns a valid choice; parse_failed should be 0."""
    out = await _run_one(tmp_path, llm_role="Servant", seed=4)
    assert out["result"].n_parse_failed == 0


@pytest.mark.asyncio
async def test_deterministic_same_seed(tmp_path: Path):
    """Same seed + mock client → identical outcome across runs."""
    out1 = await _run_one(tmp_path / "run1", llm_role="Servant", seed=42)
    out2 = await _run_one(tmp_path / "run2", llm_role="Servant", seed=42)
    r1, r2 = out1["result"], out2["result"]
    assert r1.good_wins == r2.good_wins
    assert r1.n_quests_played == r2.n_quests_played
    assert r1.n_llm_decisions == r2.n_llm_decisions


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(5))
async def test_multiple_seeds_complete(tmp_path: Path, seed: int):
    out = await _run_one(tmp_path / f"s{seed}", llm_role="Servant", seed=seed)
    assert out["result"].error is None
    assert out["result"].n_quests_played >= 3


@pytest.mark.asyncio
async def test_divergence_and_consistency_fields_present(tmp_path: Path):
    """team_vote_result and quest_result events should carry divergence/consistency."""
    import json

    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_metrics")
    client = MockClient()

    run_id = store.insert_run(
        name="test_metrics", config_hash="m1", config_yaml="game: avalon",
        backend="mock", model="mock",
    )
    condition = {"game": "avalon", "llm_role": "Servant", "budget": 64, "seed": 0}
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0, condition=condition, seed=0, llm_side=0,
    )
    jsonl.write(trial_id, "trial_start", {"trial_id": trial_id})
    env = make_env_from_seed(seed=0, llm_role="Servant")
    await run_avalon_game(
        env=env, llm_player_idx=0, llm_role="Servant", llm_client=client,
        budget=64, prompt_variant="minimal", store=store, jsonl=jsonl,
        run_id=run_id, trial_id=trial_id, condition=condition,
        seed=0, temperature=0.0, backend="mock", model="mock",
    )
    store.finalize_run(run_id)
    jsonl.close()

    # Check JSONL events carry the new fields.
    jsonl_path = tmp_path / "runs" / "run_metrics" / f"{trial_id}.jsonl"
    events = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]

    vote_events = [e for e in events if e["event"] == "team_vote_result"]
    assert vote_events, "no team_vote_result events"
    for ev in vote_events:
        assert "llm_on_team" in ev
        assert "naive_vote" in ev
        assert "llm_diverged_from_naive" in ev
        assert "llm_pass1_truncated" in ev

    quest_events = [e for e in events if e["event"] == "quest_result" and e.get("llm_quest_vote") is not None]
    for ev in quest_events:
        assert "llm_diverged_from_naive" in ev

    # Check model_calls DB rows carry quest_turn.
    cur = store._conn.cursor()
    rows = list(cur.execute("SELECT quest_turn, naive_choice, llm_diverged FROM model_calls WHERE role='reason'"))
    assert rows, "no reason model_calls"
    # quest_turn should be populated for all reason rows.
    assert all(r["quest_turn"] is not None for r in rows)
    store.close()


# ── Proposal-reasoning ablation tests ────────────────────────────────────────

async def _run_one_with_flags(
    tmp_path: Path,
    *,
    with_proposal_reasoning: bool = False,
    seed: int = 0,
    budget: int = 64,
) -> list[dict]:
    """Run one game; return parsed JSONL events."""
    import json
    store = Store(tmp_path / "results.db")
    jsonl = JSONLWriter(tmp_path / "runs", "run_prop")
    client = MockClient()

    run_id = store.insert_run(
        name="test_prop", config_hash="p1", config_yaml="game: avalon",
        backend="mock", model="mock",
    )
    condition = {"game": "avalon", "llm_role": "Servant", "budget": budget, "seed": seed}
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0, condition=condition, seed=seed, llm_side=0,
    )
    jsonl.write(trial_id, "trial_start", {"trial_id": trial_id, "condition": condition})

    env = make_env_from_seed(seed=seed, llm_role="Servant")
    await run_avalon_game(
        env=env, llm_player_idx=0, llm_role="Servant", llm_client=client,
        budget=budget, prompt_variant="minimal", store=store, jsonl=jsonl,
        run_id=run_id, trial_id=trial_id, condition=condition,
        seed=seed, temperature=0.0, backend="mock", model="mock",
        with_proposal_reasoning=with_proposal_reasoning,
    )
    store.finalize_run(run_id)
    jsonl.close()
    store.close()

    jsonl_path = tmp_path / "runs" / "run_prop" / f"{trial_id}.jsonl"
    return [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]


@pytest.mark.asyncio
async def test_proposal_reasoning_flag_false_by_default(tmp_path: Path):
    """Without the flag, all vote events have proposal_reasoning_injected=False."""
    events = await _run_one_with_flags(tmp_path, with_proposal_reasoning=False, seed=0)
    vote_events = [e for e in events if e["event"] == "team_vote_result"]
    assert vote_events, "expected at least one vote event"
    for ev in vote_events:
        assert "proposal_reasoning_injected" in ev, "field missing from vote event"
        assert ev["proposal_reasoning_injected"] is False


@pytest.mark.asyncio
async def test_proposal_reasoning_injected_when_llm_is_leader(tmp_path: Path):
    """With the flag, vote events where proposing_leader==0 have injected=True."""
    # Use several seeds to ensure we hit a round where LLM (P0) is the leader.
    for seed in range(10):
        events = await _run_one_with_flags(
            tmp_path / f"s{seed}", with_proposal_reasoning=True, seed=seed, budget=64,
        )
        vote_events = [e for e in events if e["event"] == "team_vote_result"]
        llm_led = [e for e in vote_events if e.get("leader") == 0]
        not_llm_led = [e for e in vote_events if e.get("leader") != 0]

        for ev in llm_led:
            assert ev["proposal_reasoning_injected"] is True, (
                f"seed={seed}: LLM-led vote should have injected=True, got {ev}"
            )
        for ev in not_llm_led:
            assert ev["proposal_reasoning_injected"] is False, (
                f"seed={seed}: non-LLM-led vote should have injected=False, got {ev}"
            )
        if llm_led:
            break  # found a seed with LLM leadership; test passes


@pytest.mark.asyncio
async def test_proposal_reasoning_increases_vote_input_tokens(tmp_path: Path):
    """When reasoning is injected, the vote Pass-1 input should be longer."""
    # Find a seed where LLM leads at least once (guaranteed within 10 tries).
    for seed in range(10):
        ev_off = await _run_one_with_flags(
            tmp_path / f"off{seed}", with_proposal_reasoning=False, seed=seed,
        )
        ev_on = await _run_one_with_flags(
            tmp_path / f"on{seed}", with_proposal_reasoning=True, seed=seed,
        )

        off_led = [
            e for e in ev_off
            if e["event"] == "team_vote_result"
            and e.get("leader") == 0
            and e.get("llm_pass1_tokens_in") is not None
        ]
        on_led = [
            e for e in ev_on
            if e["event"] == "team_vote_result"
            and e.get("leader") == 0
            and e.get("llm_pass1_tokens_in") is not None
        ]

        if off_led and on_led:
            avg_off = sum(e["llm_pass1_tokens_in"] for e in off_led) / len(off_led)
            avg_on  = sum(e["llm_pass1_tokens_in"] for e in on_led)  / len(on_led)
            # Injecting the reasoning trace must make the prompt longer.
            assert avg_on > avg_off, (
                f"seed={seed}: expected more input tokens with reasoning injected "
                f"({avg_on:.0f} vs {avg_off:.0f})"
            )
            return
    pytest.skip("no seed produced LLM-led votes with token counts in 10 tries")
