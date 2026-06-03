"""Avalon game runner — full game loop with two-pass LLM agent + naive bots.

``run_avalon_game()`` plays one complete Avalon game:
  - LLM (player 0 by default) uses the two-pass architecture:
      Pass 1: free reasoning up to B tokens (the experimental variable).
      Pass 2: constrained selection from the enumerated legal options
              (always produces a valid action; parse failures are logged).
  - Players 1–4 are naive bots (Light et al., 2023 strategy).

All decisions are logged to JSONL for full trace inspection.
Game outcome and model calls are written to SQLite for aggregate analysis.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cot_knob.games.avalon_summarizer import AvalonSummarizer

import numpy as np

# Bring third_party onto the path so the vendored engine can be imported.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY = _REPO_ROOT / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from avalon_llm.engine import AvalonGameEnvironment  # type: ignore[import]
from avalon_llm.avalon_exception import AvalonEnvException  # type: ignore[import]

from cot_knob.games.avalon_bots import (
    BayesianBeliefState,
    bayesian_choose_team,
    bayesian_team_vote,
    naive_choose_team,
    naive_team_vote,
    naive_quest_vote,
    naive_assassination_target,
)
from cot_knob.llm.client import LLMClient
from cot_knob.prompts.avalon import (
    PromptVariant,
    SYSTEM_PROMPT_PASS2,
    get_system_prompt,
    render_discussion_prompt,
    render_team_proposal_prompt,
    render_team_vote_prompt,
    render_quest_vote_prompt,
)
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


@dataclass
class AvalonGameResult:
    trial_id: str
    good_wins: bool
    llm_wins: bool
    llm_role: str
    llm_is_good: bool
    n_quest_attempts: int   # total team-proposal cycles (across all quests)
    n_quests_played: int    # quests actually resolved (0–5)
    n_good_quests: int      # quests won by good side
    n_llm_decisions: int    # total LLM calls (Pass-1 + Pass-2 pairs)
    n_parse_failed: int     # decisions that fell back to naive
    llm_pass1_tokens_total: int
    error: str | None = None


def make_env_from_seed(
    seed: int,
    llm_role: str,
    llm_player_idx: int = 0,
    shuffle_all_roles: bool = False,
) -> AvalonGameEnvironment:
    """Build a deterministic 5-player Avalon environment.

    Player ``llm_player_idx`` is assigned ``llm_role``. The remaining 4 roles
    are shuffled by ``seed`` while preserving the standard 5-player composition:
    Merlin + 2 Loyal Servants (good) vs. Minion + Assassin (evil).

    When ``shuffle_all_roles=True``, all 4 non-LLM roles are shuffled together so
    evil players can appear at any of P1–P4. This is the methodologically correct
    setup for Bayesian reasoning evaluation. When False (default, backward-compatible),
    evil is pinned to P3/P4 — matching the original E1 experiment structure.
    """
    rng = np.random.default_rng(seed)

    if llm_role == "Merlin":
        remaining_good = ["Servant", "Servant"]
    elif llm_role == "Servant":
        remaining_good = ["Merlin", "Servant"]
    else:
        raise ValueError(f"Unsupported LLM role for environment construction: {llm_role!r}")

    if shuffle_all_roles:
        non_llm = rng.permutation(remaining_good + ["Minion", "Assassin"]).tolist()
        role_names: list[str] = [llm_role] + non_llm
    else:
        other_good = rng.permutation(remaining_good).tolist()
        evil = rng.permutation(["Minion", "Assassin"]).tolist()
        role_names = [llm_role] + other_good + evil
    if llm_player_idx != 0:
        # Swap so LLM's role ends up at the requested index.
        role_names[0], role_names[llm_player_idx] = (
            role_names[llm_player_idx], role_names[0]
        )

    quest_leader = int(rng.integers(0, 5))
    return AvalonGameEnvironment.from_presets({
        "num_players": 5,
        "quest_leader": quest_leader,
        "role_names": role_names,
    })


async def run_avalon_game(
    *,
    env: AvalonGameEnvironment,
    llm_player_idx: int,
    llm_role: str,
    llm_client: LLMClient,
    budget: int,
    prompt_variant: PromptVariant,
    store: Store,
    jsonl: JSONLWriter,
    run_id: str,
    trial_id: str,
    condition: dict[str, Any],
    seed: int,
    temperature: float,
    backend: str,
    model: str,
    with_discussion: bool = False,
    summarizer: "Any | None" = None,
    bot_strategy: str = "naive",
    sc_samples: int = 1,
    with_proposal_reasoning: bool = False,
) -> AvalonGameResult:
    """Play one Avalon game and record all events.

    Returns an ``AvalonGameResult`` with the outcome and aggregate metrics.
    All per-decision traces are written to JSONL; model calls and the trial
    outcome are written to SQLite.
    """
    # Independent RNG for naive-bot randomness (offset avoids collision with env seed).
    rng = np.random.default_rng(seed + 999_999)
    system_prompt = get_system_prompt(llm_player_idx, llm_role, prompt_variant)

    # Bayesian belief states for Servant-role bots (one per non-LLM Servant).
    # Created fresh each game; updated after every quest result in phase 2.
    servant_beliefs: dict[int, BayesianBeliefState] = {}
    if bot_strategy == "bayesian":
        for _p in range(env.num_players):
            if _p == llm_player_idx:
                continue
            _, _rname, _ = env.get_role(_p)
            if _rname == "Servant":
                servant_beliefs[_p] = BayesianBeliefState(
                    observer=_p,
                    n_players=env.num_players,
                    n_evil=2,
                )

    # Running game history — append events here; pass to prompt renderers.
    history: list[dict[str, Any]] = []

    n_decisions = 0        # LLM decision calls (each = Pass-1 + Pass-2)
    n_parse_failed = 0
    pass1_tokens_total = 0
    n_quest_attempts = 0   # team-proposal cycles
    error: str | None = None
    # Stores the LLM's Pass-1 reasoning from the most recent proposal step so it
    # can be injected into the vote prompt when with_proposal_reasoning=True.
    _llm_proposal_reasoning: str | None = None

    jsonl.write(trial_id, "game_start", {
        "trial_id": trial_id,
        "seed": seed,
        "llm_player_idx": llm_player_idx,
        "llm_role": llm_role,
        "role_names": env.role_names,
        "quest_leader_start": env.get_quest_leader(),
        "budget": budget,
        "prompt_variant": prompt_variant,
        "with_discussion": with_discussion,
        "condition": condition,
    })

    try:
        while not env.done:
            phase_id, phase_name = env.get_phase()

            # ── Phase 0: Team Selection ────────────────────────────────────
            if phase_id == 0:
                leader = env.get_quest_leader()
                quest_turn = env.turn
                round_num = env.round
                n_quest_attempts += 1

                if leader == llm_player_idx:
                    # Compute naive team for divergence tracking (save/restore rng
                    # so that a parse-failure fallback call sees the same state).
                    _rng_st = rng.bit_generator.state
                    naive_team_fs = naive_choose_team(env, leader, rng)
                    rng.bit_generator.state = _rng_st

                    prompt, choices = render_team_proposal_prompt(
                        env, llm_player_idx, history, variant=prompt_variant
                    )
                    chosen_str, tel = await _sc_two_pass(
                        client=llm_client,
                        system=system_prompt,
                        prompt=prompt,
                        choices=choices,
                        budget=budget,
                        temperature=temperature,
                        seed=seed * 100_000 + n_decisions,
                        sc_samples=sc_samples,
                    )
                    n_decisions += 1
                    pass1_tokens_total += tel["pass1_tokens_out"]

                    if tel["parse_failed"]:
                        n_parse_failed += 1
                        team = naive_choose_team(env, leader, rng)
                    else:
                        try:
                            team = frozenset(int(x.strip()) for x in chosen_str.split(","))
                            if len(team) != env.get_team_size():
                                raise ValueError("wrong size")
                        except (ValueError, AttributeError):
                            n_parse_failed += 1
                            team = naive_choose_team(env, leader, rng)

                    naive_str = ",".join(str(p) for p in sorted(naive_team_fs))
                    diverged = (team != naive_team_fs)
                    _log_model_calls(
                        store, trial_id, tel, temperature,
                        seed=seed * 100_000 + n_decisions - 1,
                        backend=backend, model=model,
                        quest_turn=quest_turn,
                        naive_choice=naive_str,
                        llm_diverged=diverged,
                        decision_phase="team_proposal",
                    )
                    jsonl.write(trial_id, "team_proposed", {
                        "quest_turn": quest_turn,
                        "round": round_num,
                        "leader": leader,
                        "team": sorted(team),
                        "team_size": env.get_team_size(),
                        "is_llm_decision": True,
                        "naive_team": sorted(naive_team_fs),
                        "llm_diverged_from_naive": diverged,
                        **_tel_summary(tel),
                    })
                    # Persist reasoning for potential injection into the vote prompt.
                    _llm_proposal_reasoning = tel["pass1_text"] if with_proposal_reasoning else None
                else:
                    if bot_strategy == "bayesian" and leader in servant_beliefs:
                        team = bayesian_choose_team(env, leader, servant_beliefs[leader], rng)
                    else:
                        team = naive_choose_team(env, leader, rng)
                    jsonl.write(trial_id, "team_proposed", {
                        "quest_turn": quest_turn,
                        "round": round_num,
                        "leader": leader,
                        "team": sorted(team),
                        "team_size": env.get_team_size(),
                        "is_llm_decision": False,
                    })
                    _llm_proposal_reasoning = None  # LLM not the leader; nothing to inject.

                env.choose_quest_team(team, leader)

            # ── Phase 1: Team Voting ───────────────────────────────────────
            elif phase_id == 1:
                current_team = list(env.get_current_quest_team())
                quest_turn = env.turn
                current_round = env.round
                # The leader who proposed is the one just before quest_leader advanced.
                proposing_leader = (env.quest_leader - 1) % env.num_players

                # ── Discussion phase (optional) ────────────────────────────
                if with_discussion:
                    disc_prompt = render_discussion_prompt(
                        env, llm_player_idx, history, current_team,
                        proposing_leader=proposing_leader,
                        variant=prompt_variant,
                    )
                    disc_comp = await llm_client.generate(
                        disc_prompt,
                        max_tokens=budget,
                        temperature=temperature,
                        seed=seed * 100_000 + n_decisions,
                        system=system_prompt,
                    )
                    n_decisions += 1
                    pass1_tokens_total += disc_comp.n_output_tokens

                    _log_model_calls(
                        store, trial_id,
                        {
                            "pass1_prompt": disc_prompt,
                            "pass1_text": disc_comp.text,
                            "pass1_tokens_in": disc_comp.n_input_tokens,
                            "pass1_tokens_out": disc_comp.n_output_tokens,
                            "pass1_finish": disc_comp.finish_reason,
                            "pass1_latency_ms": disc_comp.latency_ms,
                            "pass2_choice": "",
                            "pass2_tokens_in": 0,
                            "pass2_tokens_out": 0,
                            "pass2_finish": "skipped",
                            "pass2_latency_ms": 0.0,
                            "parse_failed": False,
                        },
                        temperature,
                        seed=seed * 100_000 + n_decisions - 1,
                        backend=backend, model=model,
                        override_role="discuss",
                        quest_turn=quest_turn,
                        decision_phase="discussion",
                    )

                    disc_ev: dict[str, Any] = {
                        "type": "discussion",
                        "quest_turn": quest_turn,
                        "round": current_round,
                        "leader": proposing_leader,
                        "team": sorted(current_team),
                        "statements": {str(llm_player_idx): disc_comp.text},
                    }
                    history.append(disc_ev)
                    jsonl.write(trial_id, "discussion", {
                        **disc_ev,
                        "llm_player_idx": llm_player_idx,
                        "llm_disc_tokens_in": disc_comp.n_input_tokens,
                        "llm_disc_tokens_out": disc_comp.n_output_tokens,
                        "llm_disc_finish": disc_comp.finish_reason,
                        "llm_disc_latency_ms": disc_comp.latency_ms,
                    })

                # Naive vote is a pure function; compute once for divergence tracking.
                naive_vote_val = naive_team_vote(env, llm_player_idx)
                naive_vote_str = "approve" if naive_vote_val == 1 else "reject"
                llm_on_team = llm_player_idx in current_team

                votes: list[int] = []
                llm_vote: int | None = None
                llm_tel: dict | None = None
                llm_vote_diverged: bool | None = None

                # Inject the LLM's own proposal reasoning into the vote prompt when
                # the LLM was the leader (proposal_reasoning is None otherwise).
                _vote_proposal_ctx = (
                    _llm_proposal_reasoning
                    if (with_proposal_reasoning and proposing_leader == llm_player_idx)
                    else None
                )

                for p in range(env.num_players):
                    if p == llm_player_idx:
                        prompt, choices = render_team_vote_prompt(
                            env, llm_player_idx, history, current_team,
                            proposing_leader=proposing_leader,
                            variant=prompt_variant,
                            proposal_reasoning=_vote_proposal_ctx,
                        )
                        vote_str, tel = await _sc_two_pass(
                            client=llm_client,
                            system=system_prompt,
                            prompt=prompt,
                            choices=choices,
                            budget=budget,
                            temperature=temperature,
                            seed=seed * 100_000 + n_decisions,
                            sc_samples=sc_samples,
                        )
                        n_decisions += 1
                        pass1_tokens_total += tel["pass1_tokens_out"]

                        if tel["parse_failed"] or vote_str not in ("approve", "reject"):
                            n_parse_failed += 1
                            vote_val = naive_team_vote(env, p)
                        else:
                            vote_val = 1 if vote_str == "approve" else 0

                        llm_vote = vote_val
                        llm_tel = tel
                        llm_vote_diverged = (vote_val != naive_vote_val)
                        _log_model_calls(
                            store, trial_id, tel, temperature,
                            seed=seed * 100_000 + n_decisions - 1,
                            backend=backend, model=model,
                            quest_turn=quest_turn,
                            naive_choice=naive_vote_str,
                            llm_diverged=llm_vote_diverged,
                            decision_phase="team_vote",
                        )
                    else:
                        if bot_strategy == "bayesian" and p in servant_beliefs:
                            vote_val = bayesian_team_vote(env, p, servant_beliefs[p])
                        else:
                            vote_val = naive_team_vote(env, p)

                    votes.append(vote_val)

                _, _, accepted = env.gather_team_votes(votes)

                ev: dict[str, Any] = {
                    "type": "team_vote_result",
                    "quest_turn": quest_turn,
                    "round": current_round,
                    "leader": proposing_leader,
                    "team": sorted(current_team),
                    "votes": votes,
                    "accepted": accepted,
                }
                history.append(ev)

                jsonl.write(trial_id, "team_vote_result", {
                    **ev,
                    "llm_player_idx": llm_player_idx,
                    "llm_vote": llm_vote,
                    "llm_on_team": llm_on_team,
                    "naive_vote": naive_vote_str,
                    "llm_diverged_from_naive": llm_vote_diverged,
                    "proposal_reasoning_injected": _vote_proposal_ctx is not None,
                    **({"llm_pass1_text": llm_tel["pass1_text"], **_tel_summary(llm_tel)} if llm_tel else {}),
                })

            # ── Phase 2: Quest Voting ──────────────────────────────────────
            elif phase_id == 2:
                current_team = sorted(env.get_current_quest_team())
                quest_turn = env.turn

                vote_map: dict[int, int] = {}
                llm_quest_vote: int | None = None
                llm_quest_tel: dict | None = None
                llm_quest_diverged: bool | None = None

                for p in current_team:
                    if p == llm_player_idx:
                        # Naive quest vote is pure (no rng).
                        naive_qvote_val = naive_quest_vote(env, p)
                        naive_qvote_str = "success" if naive_qvote_val == 1 else "fail"

                        prompt, choices = render_quest_vote_prompt(
                            env, llm_player_idx, history, current_team,
                            variant=prompt_variant,
                        )
                        vote_str, tel = await _sc_two_pass(
                            client=llm_client,
                            system=system_prompt,
                            prompt=prompt,
                            choices=choices,
                            budget=budget,
                            temperature=temperature,
                            seed=seed * 100_000 + n_decisions,
                            sc_samples=sc_samples,
                        )
                        n_decisions += 1
                        pass1_tokens_total += tel["pass1_tokens_out"]

                        if tel["parse_failed"] or vote_str not in ("success", "fail"):
                            n_parse_failed += 1
                            vote_val = naive_quest_vote(env, p)
                        else:
                            vote_val = 1 if vote_str == "success" else 0

                        llm_quest_vote = vote_val
                        llm_quest_tel = tel
                        llm_quest_diverged = (vote_val != naive_qvote_val)
                        _log_model_calls(
                            store, trial_id, tel, temperature,
                            seed=seed * 100_000 + n_decisions - 1,
                            backend=backend, model=model,
                            quest_turn=quest_turn,
                            naive_choice=naive_qvote_str,
                            llm_diverged=llm_quest_diverged,
                            decision_phase="quest_vote",
                        )
                    else:
                        vote_val = naive_quest_vote(env, p)

                    vote_map[p] = vote_val

                # Order votes to match team order (engine just counts, but log for clarity).
                quest_votes = [vote_map[p] for p in current_team]
                _, _, succeeded, num_fails = env.gather_quest_votes(quest_votes)

                # Bayesian update: all Servant bots refine their evil beliefs.
                if servant_beliefs:
                    for _belief in servant_beliefs.values():
                        _belief.update(frozenset(current_team), succeeded)

                ev = {
                    "type": "quest_result",
                    "quest_turn": quest_turn,
                    "team": current_team,
                    "vote_map": vote_map,
                    "succeeded": succeeded,
                    "num_fails": num_fails,
                }
                history.append(ev)

                jsonl.write(trial_id, "quest_result", {
                    **ev,
                    "llm_player_idx": llm_player_idx,
                    "llm_quest_vote": llm_quest_vote,
                    "llm_diverged_from_naive": llm_quest_diverged,
                    **(
                        {"llm_pass1_text": llm_quest_tel["pass1_text"], **_tel_summary(llm_quest_tel)}
                        if llm_quest_tel else {}
                    ),
                })

                # ── Post-quest summary (optional Sonnet call) ──────────────
                if summarizer is not None:
                    await summarizer.summarize(
                        trial_id=trial_id,
                        after_quest_idx=quest_turn,
                        history=history,
                        quest_results=list(env.quest_results),
                        llm_player_idx=llm_player_idx,
                        llm_role=llm_role,
                        store=store,
                        jsonl=jsonl,
                    )

            # ── Phase 3: Assassination ─────────────────────────────────────
            elif phase_id == 3:
                assassin_idx = env.get_assassin()
                merlin_idx = next(
                    i for i in range(env.num_players)
                    if env.get_role(i)[1] == "Merlin"
                )
                # LLM assassination not planned for Servant/Merlin cells; naive always.
                target = naive_assassination_target(env, rng)
                _, _, good_wins_final = env.choose_assassination_target(assassin_idx, target)
                target_is_merlin = (target == merlin_idx)

                jsonl.write(trial_id, "assassination", {
                    "assassin_idx": assassin_idx,
                    "merlin_idx": merlin_idx,
                    "target": target,
                    "target_is_merlin": target_is_merlin,
                    "good_wins": good_wins_final,
                })

            else:
                raise ValueError(f"Unknown phase id: {phase_id}")

    except AvalonEnvException as exc:
        error = f"AvalonEnvException: {exc}"
        jsonl.write(trial_id, "game_error", {"error": error})
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        jsonl.write(trial_id, "game_error", {"error": error})

    # ── Outcome ───────────────────────────────────────────────────────────
    good_wins = bool(env.good_victory)
    _role_id, _role_name, llm_is_good = env.get_role(llm_player_idx)
    llm_wins = (good_wins and llm_is_good) or (not good_wins and not llm_is_good)

    n_quests_played = len(env.quest_results)
    n_good_quests = int(sum(env.quest_results))
    winner_str = "error" if error else ("llm" if llm_wins else "bots")

    store.finalize_trial(
        trial_id,
        winner=winner_str,
        n_turns_total=n_quest_attempts,
        n_llm_turns=n_decisions,
        n_uct_turns=0,
        final_score_llm=(n_good_quests if llm_is_good else n_quests_played - n_good_quests),
        final_score_uct=0,
        n_tokens_llm_total=pass1_tokens_total,
        error=error,
    )

    jsonl.write(trial_id, "game_end", {
        "good_wins": good_wins,
        "llm_wins": llm_wins,
        "winner": winner_str,
        "quest_results": [bool(r) for r in env.quest_results],
        "n_quest_attempts": n_quest_attempts,
        "n_quests_played": n_quests_played,
        "n_good_quests": n_good_quests,
        "n_llm_decisions": n_decisions,
        "n_parse_failed": n_parse_failed,
        "llm_pass1_tokens_total": pass1_tokens_total,
        "error": error,
    })

    return AvalonGameResult(
        trial_id=trial_id,
        good_wins=good_wins,
        llm_wins=llm_wins,
        llm_role=llm_role,
        llm_is_good=llm_is_good,
        n_quest_attempts=n_quest_attempts,
        n_quests_played=n_quests_played,
        n_good_quests=n_good_quests,
        n_llm_decisions=n_decisions,
        n_parse_failed=n_parse_failed,
        llm_pass1_tokens_total=pass1_tokens_total,
        error=error,
    )


# ── Two-pass helper ────────────────────────────────────────────────────────

async def _sc_two_pass(
    *,
    client: LLMClient,
    system: str,
    prompt: str,
    choices: list[str],
    budget: int,
    temperature: float,
    seed: int,
    sc_samples: int,
) -> tuple[str, dict[str, Any]]:
    """Self-consistency wrapper: run _two_pass sc_samples times, return plurality.

    With sc_samples=1 this is a no-op (directly calls _two_pass).
    Token count in telemetry reflects total across all K samples so that
    compute comparisons remain valid (K×B tokens ~ single B×K call).
    """
    if sc_samples <= 1:
        return await _two_pass(
            client=client, system=system, prompt=prompt,
            choices=choices, budget=budget, temperature=temperature, seed=seed,
        )

    results: list[tuple[str, dict[str, Any]]] = []
    for k in range(sc_samples):
        chosen, tel = await _two_pass(
            client=client, system=system, prompt=prompt,
            choices=choices, budget=budget, temperature=temperature,
            seed=seed + k * 1_000_000,
        )
        results.append((chosen, tel))

    counts: Counter[str] = Counter(chosen for chosen, _ in results)
    plurality = counts.most_common(1)[0][0]

    first_tel = results[0][1]
    merged_tel = {**first_tel}
    merged_tel["pass1_tokens_out"] = sum(tel["pass1_tokens_out"] for _, tel in results)
    merged_tel["sc_samples"] = sc_samples
    merged_tel["sc_choices"] = [chosen for chosen, _ in results]

    return plurality, merged_tel


async def _two_pass(
    *,
    client: LLMClient,
    system: str,
    prompt: str,
    choices: list[str],
    budget: int,
    temperature: float,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    """Run Pass-1 (free reasoning) + Pass-2 (constrained choice).

    Returns ``(chosen_str, telemetry_dict)``.
    ``chosen_str`` is always one element of ``choices``; ``parse_failed``
    in the telemetry is True if the model's output didn't match cleanly.
    """
    parse_failed = False

    if budget > 0:
        comp = await client.generate(
            prompt,
            max_tokens=budget,
            temperature=temperature,
            seed=seed,
            system=system,
        )
        pass1_text = comp.text
        pass1_tokens_in = comp.n_input_tokens
        pass1_tokens_out = comp.n_output_tokens
        pass1_finish = comp.finish_reason
        pass1_latency = comp.latency_ms
    else:
        pass1_text = ""
        pass1_tokens_in = 0
        pass1_tokens_out = 0
        pass1_finish = "skipped"
        pass1_latency = 0.0

    # Pass-2: include Pass-1 reasoning as context then constrain to choices.
    pass2_prompt = (
        f"{prompt}\n\n"
        f"Your reasoning:\n{pass1_text}\n\n"
        f"Choose exactly one:\n"
        + "\n".join(f"- {c}" for c in choices)
    )

    choice = await client.generate_choice(
        pass2_prompt,
        choices=choices,
        temperature=temperature,
        seed=seed,
        system=SYSTEM_PROMPT_PASS2,
    )
    if choice.raw.get("__choice_parse_failed"):
        parse_failed = True

    # Proxy for whether the constrained decoder had to "fix" the model's output:
    # True when no valid choice string appears anywhere in the pass-1 reasoning,
    # meaning the constraint likely forced a choice the model didn't express.
    text_lower = pass1_text.lower()
    pass2_constraint_needed = (
        None if not pass1_text  # B=0 or skipped — not applicable
        else not any(c.lower() in text_lower for c in choices)
    )

    return choice.choice_text, {
        "pass1_prompt": prompt,
        "pass1_text": pass1_text,
        "pass1_tokens_in": pass1_tokens_in,
        "pass1_tokens_out": pass1_tokens_out,
        "pass1_finish": pass1_finish,
        "pass1_latency_ms": pass1_latency,
        "pass2_choice": choice.choice_text,
        "pass2_tokens_in": choice.n_input_tokens,
        "pass2_tokens_out": choice.n_output_tokens,
        "pass2_finish": choice.finish_reason,
        "pass2_latency_ms": choice.latency_ms,
        "parse_failed": parse_failed,
        "pass2_constraint_needed": pass2_constraint_needed,
    }



def _log_model_calls(
    store: Store,
    trial_id: str,
    tel: dict[str, Any],
    temperature: float,
    *,
    seed: int,
    backend: str,
    model: str,
    override_role: str | None = None,
    quest_turn: int | None = None,
    naive_choice: str | None = None,
    llm_diverged: bool | None = None,
    decision_phase: str | None = None,
) -> None:
    pass1_role = override_role or "reason"
    constraint_needed = tel.get("pass2_constraint_needed")
    if tel["pass1_finish"] != "skipped":
        store.insert_model_call(
            trial_id=trial_id, turn_id=None, role=pass1_role,
            prompt_text=tel["pass1_prompt"],
            response_text=tel["pass1_text"],
            n_input_tokens=tel["pass1_tokens_in"],
            n_output_tokens=tel["pass1_tokens_out"],
            finish_reason=tel["pass1_finish"],
            temperature=temperature,
            seed=seed,
            latency_ms=tel["pass1_latency_ms"],
            backend=backend, model=model,
            quest_turn=quest_turn,
            naive_choice=naive_choice,
            llm_diverged=llm_diverged,
            pass2_constraint_needed=constraint_needed,
            decision_phase=decision_phase,
        )
    if tel["pass2_finish"] != "skipped":
        store.insert_model_call(
            trial_id=trial_id, turn_id=None, role="select",
            prompt_text="",
            response_text=tel["pass2_choice"],
            n_input_tokens=tel["pass2_tokens_in"],
            n_output_tokens=tel["pass2_tokens_out"],
            finish_reason=tel["pass2_finish"],
            temperature=temperature,
            seed=seed,
            latency_ms=tel["pass2_latency_ms"],
            backend=backend, model=model,
            quest_turn=quest_turn,
            naive_choice=naive_choice,
            llm_diverged=llm_diverged,
            pass2_constraint_needed=constraint_needed,
            decision_phase=decision_phase,
        )


def _tel_summary(tel: dict[str, Any]) -> dict[str, Any]:
    """Compact telemetry subset for JSONL (excludes long pass1_text)."""
    return {
        "llm_pass1_tokens_in": tel["pass1_tokens_in"],
        "llm_pass1_tokens_out": tel["pass1_tokens_out"],
        "llm_pass1_finish": tel["pass1_finish"],
        "llm_pass1_truncated": tel["pass1_finish"] == "length",  # budget was binding
        "llm_pass1_latency_ms": tel["pass1_latency_ms"],
        "llm_pass2_choice": tel["pass2_choice"],
        "llm_pass2_tokens_out": tel["pass2_tokens_out"],
        "llm_pass2_finish": tel["pass2_finish"],
        "llm_pass2_latency_ms": tel["pass2_latency_ms"],
        "llm_parse_failed": tel["parse_failed"],
        "llm_pass2_constraint_needed": tel.get("pass2_constraint_needed"),
    }
