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
from cot_knob.games.reversi import ReversiState
from cot_knob.games.reversi import initial_state as reversi_initial_state
from cot_knob.memory.base import TurnRecord
from cot_knob.memory.summary import StructuredSummaryMemory
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store

# --- Phase tagging --------------------------------------------------------------
# Default cutoffs for Reversi (60-ply game). Override via play_match params
# for shorter games (Nim: early_cutoff=4, late_after=12).
EARLY_CUTOFF = 10
LATE_AFTER = 50


def _phase_for_turn(turn_idx: int, *, early_cutoff: int, late_after: int) -> str:
    if turn_idx < early_cutoff:
        return "early"
    if turn_idx >= late_after:
        return "late"
    return "mid"


def _score_state(state: GameState, llm_side: int) -> tuple[int, int, int]:
    """Compute (winner_int, final_score_llm, final_score_uct) at safety-cap.

    Tries to count ``state.board`` (Reversi) and falls back to a draw for
    games without a board attribute (Nim always terminates before cap).
    """
    try:
        board = state.board  # type: ignore[attr-defined]
        score_b = sum(1 for row in board for v in row if v == 1)
        score_w = sum(1 for row in board for v in row if v == -1)
    except AttributeError:
        # Game type has no board (e.g. Nim). Declare a draw.
        return 0, 0, 0
    winner_int = 1 if score_b > score_w else (-1 if score_w > score_b else 0)
    final_llm = score_b if llm_side == 1 else score_w
    final_uct = score_w if llm_side == 1 else score_b
    return winner_int, final_llm, final_uct


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
    opp_kind: str = "uct",  # label for the opponent agent in JSONL / DB
    oracle_iters: int = 2000,
    oracle_agent: Agent | None = None,
    initial: GameState | None = None,
    max_turns_safety: int = 200,
    backend: str = "mock",
    model: str = "mock",
    early_cutoff: int = EARLY_CUTOFF,
    late_after: int = LATE_AFTER,
) -> TrialResult:
    """Play one game; returns a TrialResult and writes everything to the store.

    ``oracle_iters`` controls the UCT used **only** for move-quality evaluation
    (``move_quality`` and ``move_regret``).  It is independent of
    ``uct_agent.iterations`` so you can play against a weak opponent (e.g.
    UCT-10) while still using a strong oracle (UCT-2000) for regret estimates.

    Pass ``oracle_agent`` explicitly to use a non-UCT oracle (e.g.
    ``NimOptimalAgent`` for Nim sweeps).
    """
    state: GameState = initial if initial is not None else reversi_initial_state()

    # Dedicated oracle — independent of the game opponent.  top_k=None returns
    # ALL legal moves with win rates for continuous regret estimation.
    if oracle_agent is None:
        oracle_agent = UCTAgent(iterations=oracle_iters, top_k=None)

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
            phase = _phase_for_turn(n_turns, early_cutoff=early_cutoff, late_after=late_after)

            # Query the oracle for every turn (both LLM and UCT) so we can
            # compare move quality on the same axis.  oracle_agent has top_k=None
            # → returns ALL legal moves ranked by win rate.  seed=42 keeps the
            # oracle evaluation reproducible across repeated experiments.
            uct_top3 = tel.uct_top3 or []
            move_quality: int | None = None
            move_regret: float | None = None
            oracle_chosen_rank: int | None = None
            n_legal = len(tel.legal_moves)

            ref = await oracle_agent.choose(state, seed=42)
            uct_top3 = ref.uct_top3 or []  # full ranking, not just top-3

            if uct_top3:
                # Binary quality: is the chosen move in the top-3?
                top3_set = {entry["move"] for entry in uct_top3[:3]}
                move_quality = 1 if move_str in top3_set else 0

                # Continuous regret: best_win_rate − chosen_win_rate.
                # Both from the mover's perspective (UCTAgent convention).
                best_wr = uct_top3[0]["win_rate"]  # already sorted best-first
                for rank_idx, entry in enumerate(uct_top3, 1):
                    if entry["move"] == move_str:
                        oracle_chosen_rank = rank_idx
                        move_regret = round(best_wr - entry["win_rate"], 4)
                        break
                # If chosen move wasn't visited (extremely rare at 2000 iters),
                # rank = last place and regret remains None rather than misleading.

            agent_kind = "llm" if agent is llm_agent else opp_kind

            turn_id = store.insert_turn(
                trial_id=trial_id, turn_idx=n_turns, phase=phase,
                player=state.current_player, agent_kind=agent_kind,
                board_state=tel.state_serialized,
                legal_moves=[state.move_to_str(m) for m in tel.legal_moves],
                chosen_move=move_str, chosen_move_id=move_id if move_id >= 0 else None,
                uct_top3=uct_top3, move_quality=move_quality,
                move_regret=move_regret,
                oracle_iters_used=oracle_iters,
                n_legal_moves=n_legal,
                oracle_chosen_rank=oracle_chosen_rank,
                latency_ms_total=latency_total_ms,
                parse_failed=tel.llm_pass2_parse_failed,
            )

            # Record model_calls (only LLM agent produces them in the smoke
            # sweep; FillerAgent will produce role='filler' rows later).
            if agent is llm_agent:
                if tel.llm_pass1_finish != "skipped":
                    store.insert_model_call(
                        trial_id=trial_id, turn_id=turn_id, role="reason",
                        prompt_text=tel.llm_pass1_prompt,
                        response_text=tel.llm_pass1_text,
                        n_input_tokens=tel.llm_pass1_tokens_in,
                        n_output_tokens=tel.llm_pass1_tokens_out,
                        finish_reason=tel.llm_pass1_finish,
                        temperature=condition.get("temperature", 0.0),
                        seed=seed,
                        latency_ms=tel.llm_pass1_latency_ms,
                        backend=backend, model=model,
                    )
                if tel.llm_pass2_finish not in ("skipped", ""):
                    store.insert_model_call(
                        trial_id=trial_id, turn_id=turn_id, role="select",
                        prompt_text="",
                        response_text=tel.llm_pass2_choice_text,
                        n_input_tokens=tel.extra.get("pass2_prompt_tokens") or 0,
                        n_output_tokens=tel.llm_pass2_tokens_out or 1,
                        finish_reason=tel.llm_pass2_finish or "unknown",
                        temperature=condition.get("temperature", 0.0),
                        seed=seed,
                        latency_ms=tel.llm_pass2_latency_ms,
                        backend=backend, model=model,
                    )

            turn_payload = {
                "turn_idx": n_turns, "phase": phase,
                "agent": agent_kind, "player": state.current_player,
                "chosen_move": move_str,
                "n_legal_moves": n_legal,
                "uct_top3": uct_top3,
                "move_quality": move_quality,
                "move_regret": move_regret,
                "oracle_chosen_rank": oracle_chosen_rank,
                "oracle_iters_used": oracle_iters,
                "latency_ms": latency_total_ms,
            }
            if agent is llm_agent:
                turn_payload.update({
                    "budget_B": condition.get("budget"),
                    "pass1_system": tel.llm_pass1_system,
                    "pass1_prompt": tel.llm_pass1_prompt,
                    "pass1_text": tel.llm_pass1_text,
                    "pass1_tokens_in": tel.llm_pass1_tokens_in,
                    "pass1_tokens_out": tel.llm_pass1_tokens_out,
                    "pass1_finish_reason": tel.llm_pass1_finish,
                    "pass1_latency_ms": tel.llm_pass1_latency_ms,
                    "pass2_finish_reason": tel.llm_pass2_finish,
                    "pass2_tokens_out": tel.llm_pass2_tokens_out,
                    "pass2_prompt_tokens": tel.extra.get("pass2_prompt_tokens"),
                    "pass2_cached_tokens": tel.extra.get("pass2_cached_tokens"),
                    "pass2_latency_ms": tel.llm_pass2_latency_ms,
                    "parse_failed": tel.llm_pass2_parse_failed,
                    "pass2_choice_text": tel.llm_pass2_choice_text,
                })
            jsonl.write(trial_id, "turn", turn_payload)

            # Apply the move FIRST so memory records the post-move state.
            # tel.state_serialized (pre-move) is already written to the DB above.
            player_this_turn = state.current_player
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

            # Update memory with the POST-MOVE state so the next prompt sees
            # correct pile sizes in the "Piles after:" line.
            if isinstance(llm_agent, LLMAgent):
                rec = TurnRecord(
                    turn_idx=n_turns - 1, player=player_this_turn,
                    move_str=move_str, state_after_serialized=state.to_serializable(),
                )
                llm_agent._memory.update(rec)  # noqa: SLF001 (intentional)
                if isinstance(llm_agent._memory, StructuredSummaryMemory):  # noqa: SLF001
                    snap = await llm_agent._memory.render()  # noqa: SLF001
                    store.insert_summary(
                        trial_id=trial_id, after_turn_idx=n_turns - 1,
                        summary_text=snap.text, n_tokens_est=snap.n_tokens_estimate,
                        kind=snap.kind,
                    )

        # Decide outcome.
        winner_int = state.winner()
        if winner_int is None:
            # Reached safety cap — fall back to board counting (Reversi) or draw.
            winner_int, final_llm, final_uct = _score_state(state, llm_side)
        else:
            _, final_llm, final_uct = _score_state(state, llm_side)
        if winner_int == 0:
            winner_str = "draw"
        else:
            winner_str = "llm" if winner_int == llm_side else opp_kind

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
