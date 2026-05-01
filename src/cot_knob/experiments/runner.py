"""Single-game runner.

``play_match(...)`` runs one Reversi game between an LLM agent (one side)
and a UCT agent (the other side), recording every turn, model call, and
summary into the supplied ``Store`` and ``JSONLWriter``.

The runner is intentionally agent-agnostic at its core: it asks each
agent ``choose(state)`` and records the returned ``TurnTelemetry``.
That's the contract that lets us drop in ``FillerAgent`` or another
LLM size later without touching this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.agents.llm_agent import LLMAgent
from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.games.base import GameState
from cot_knob.games.reversi import ReversiState, initial_state
from cot_knob.memory.base import TurnRecord
from cot_knob.memory.summary import StructuredSummaryMemory
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store

# --- Phase tagging --------------------------------------------------------------
# The proposal asks for "early" (turns 1-10), "mid", "late" (last 10).
# At rendering time we don't know the final turn count yet, so we tag based on
# turn index alone using a conservative heuristic and post-hoc correct only
# the "late" tag in analytics if needed. For Reversi 8x8, total turns is
# typically ~58-62 so this is close enough for the smoke sweep.
EARLY_CUTOFF = 10
LATE_AFTER = 50


def _phase_for_turn(turn_idx: int) -> str:
    if turn_idx < EARLY_CUTOFF:
        return "early"
    if turn_idx >= LATE_AFTER:
        return "late"
    return "mid"


@dataclass
class TrialResult:
    trial_id: str
    winner: str  # "llm" | "uct" | "draw" | "error"
    n_turns: int
    n_llm_turns: int
    n_uct_turns: int
    final_score_llm: int
    final_score_uct: int
    error: str | None = None


async def _agent_for_player(
    player: int,
    *,
    llm_side: int,
    llm_agent: Agent,
    uct_agent: Agent,
) -> Agent:
    return llm_agent if player == llm_side else uct_agent


async def play_match(
    *,
    store: Store,
    jsonl: JSONLWriter,
    run_id: str,
    cell_index: int,
    condition: dict[str, Any],
    seed: int,
    llm_side: int,
    llm_agent: Agent,
    uct_agent: Agent,
    initial: GameState | None = None,
    max_turns_safety: int = 200,
    backend: str = "mock",
    model: str = "mock",
) -> TrialResult:
    """Play one game; returns a TrialResult and writes everything to the store."""
    state: ReversiState = initial if isinstance(initial, ReversiState) else initial_state()

    trial_id = store.insert_trial(
        run_id=run_id, cell_index=cell_index, condition=condition,
        seed=seed, llm_side=llm_side,
    )
    jsonl.write(trial_id, "trial_start", {"trial_id": trial_id, "condition": condition,
                                          "seed": seed, "llm_side": llm_side})

    n_turns = 0
    n_llm = 0
    n_uct = 0
    pass_streak = 0
    error: str | None = None

    try:
        while not state.is_terminal() and n_turns < max_turns_safety and pass_streak < 2:
            agent = await _agent_for_player(
                state.current_player, llm_side=llm_side,
                llm_agent=llm_agent, uct_agent=uct_agent,
            )
            t0 = time.perf_counter()
            tel: TurnTelemetry = await agent.choose(state, seed=seed * 1000 + n_turns)
            latency_total_ms = (time.perf_counter() - t0) * 1000.0

            move_id = tel.chosen_move
            move_str = state.move_to_str(move_id) if move_id >= 0 else "pass"
            phase = _phase_for_turn(n_turns)

            # If LLM moved: ask UCT for a top-3 oracle for move quality logging.
            uct_top3 = tel.uct_top3 or []
            move_quality: int | None = None
            if agent is llm_agent and isinstance(uct_agent, UCTAgent):
                # Ask the same UCT to rank moves at the *current* state so we
                # have an oracle ranking. Use a fresh deterministic seed so
                # repeated experiments are comparable.
                ref = await uct_agent.choose(state, seed=42)
                uct_top3 = ref.uct_top3 or []
                top3_set = {entry["move"] for entry in uct_top3}
                if move_str in top3_set:
                    move_quality = 1
                elif uct_top3:
                    move_quality = 0

            agent_kind = "llm" if agent is llm_agent else "uct"

            turn_id = store.insert_turn(
                trial_id=trial_id, turn_idx=n_turns, phase=phase,
                player=state.current_player, agent_kind=agent_kind,
                board_state=tel.state_serialized,
                legal_moves=[state.move_to_str(m) for m in tel.legal_moves],
                chosen_move=move_str, chosen_move_id=move_id if move_id >= 0 else None,
                uct_top3=uct_top3, move_quality=move_quality,
                latency_ms_total=latency_total_ms,
                parse_failed=tel.llm_pass2_parse_failed,
            )

            # Record model_calls (only LLM agent produces them in the smoke
            # sweep; FillerAgent will produce role='filler' rows later).
            if agent is llm_agent:
                if tel.llm_pass1_finish != "skipped":
                    store.insert_model_call(
                        trial_id=trial_id, turn_id=turn_id, role="reason",
                        prompt_text="<reason_prompt>",
                        response_text=tel.llm_pass1_text,
                        n_input_tokens=tel.llm_pass1_tokens_in,
                        n_output_tokens=tel.llm_pass1_tokens_out,
                        finish_reason=tel.llm_pass1_finish,
                        temperature=condition.get("temperature", 0.0),
                        seed=seed,
                        latency_ms=tel.llm_pass1_latency_ms,
                        backend=backend, model=model,
                    )
                store.insert_model_call(
                    trial_id=trial_id, turn_id=turn_id, role="select",
                    prompt_text="<select_prompt>",
                    response_text=tel.llm_pass2_choice_text,
                    n_input_tokens=0, n_output_tokens=1,
                    finish_reason="stop",
                    temperature=condition.get("temperature", 0.0),
                    seed=seed,
                    latency_ms=tel.llm_pass2_latency_ms,
                    backend=backend, model=model,
                )

            jsonl.write(trial_id, "turn", {
                "turn_idx": n_turns, "phase": phase,
                "agent": agent_kind, "player": state.current_player,
                "chosen_move": move_str,
                "uct_top3": uct_top3,
                "move_quality": move_quality,
                "pass1_text": tel.llm_pass1_text,
                "pass1_tokens_out": tel.llm_pass1_tokens_out,
                "latency_ms": latency_total_ms,
            })

            # Update memory + write a summary snapshot when the LLM is the
            # one with structured-summary memory.
            if isinstance(llm_agent, LLMAgent):
                rec = TurnRecord(
                    turn_idx=n_turns, player=state.current_player,
                    move_str=move_str, state_after_serialized=tel.state_serialized,
                )
                llm_agent._memory.update(rec)  # noqa: SLF001 (intentional)
                if isinstance(llm_agent._memory, StructuredSummaryMemory):  # noqa: SLF001
                    snap = await llm_agent._memory.render()  # noqa: SLF001
                    store.insert_summary(
                        trial_id=trial_id, after_turn_idx=n_turns,
                        summary_text=snap.text, n_tokens_est=snap.n_tokens_estimate,
                        kind=snap.kind,
                    )

            # Apply the move (or pass).
            if move_id < 0:
                pass_streak += 1
                state = state.apply_move(-1)
            else:
                pass_streak = 0
                state = state.apply_move(move_id)

            n_turns += 1
            if agent is llm_agent:
                n_llm += 1
            else:
                n_uct += 1

        # Decide outcome.
        winner_int = state.winner()
        if winner_int is None:
            # Reached safety cap.
            score_b = sum(1 for row in state.board for v in row if v == 1)
            score_w = sum(1 for row in state.board for v in row if v == -1)
            winner_int = 1 if score_b > score_w else (-1 if score_w > score_b else 0)
        score_b = sum(1 for row in state.board for v in row if v == 1)
        score_w = sum(1 for row in state.board for v in row if v == -1)
        final_llm = score_b if llm_side == 1 else score_w
        final_uct = score_w if llm_side == 1 else score_b
        if winner_int == 0:
            winner_str = "draw"
        else:
            winner_str = "llm" if winner_int == llm_side else "uct"

    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        winner_str = "error"
        final_llm = -1
        final_uct = -1

    store.finalize_trial(
        trial_id, winner=winner_str, n_turns_total=n_turns,
        n_llm_turns=n_llm, n_uct_turns=n_uct,
        final_score_llm=final_llm, final_score_uct=final_uct,
        error=error,
    )
    jsonl.write(trial_id, "trial_end", {
        "winner": winner_str, "n_turns": n_turns,
        "score_llm": final_llm, "score_uct": final_uct,
        "error": error,
    })
    return TrialResult(
        trial_id=trial_id, winner=winner_str, n_turns=n_turns,
        n_llm_turns=n_llm, n_uct_turns=n_uct,
        final_score_llm=final_llm, final_score_uct=final_uct,
        error=error,
    )
