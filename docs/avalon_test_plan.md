# Avalon Test Plan — AvalonBench Integration

**Audience:** Self / future me, planning the AvalonBench experiment phase that follows the Nim pilot.
**Status:** Design doc. Nothing has been built or run yet.
**Companion docs:** [`nim_experiment_findings.md`](nim_experiment_findings.md) (lessons from the Nim pilot), [`nim_test_backlog.md`](nim_test_backlog.md) (parallel high-N Nim work), [`pc_dev_setup.md`](pc_dev_setup.md) (rig).

---

## 1. Why Avalon, why now

The Nim pilot showed that on a game where the optimal policy has a closed-form symbolic solution (XOR / nim-sum), CoT budget is **not a clean difficulty knob** at small model scale: Llama 3.1 8B sits at chance level across all budgets and prompt structures we tested, because it cannot reliably execute XOR + the inverse mapping. More tokens give the model more rope to commit to wrong calculations.

This is consistent with the underlying hypothesis but ruled out the planned game-of-record: Reversi via Ludii is gated on a model that can derive board legality from an ASCII grid, and the [OOD probe](ood_probe_findings.md) showed R1-Distill-7B cannot. Llama 8B on Reversi is plausibly better but untested.

**Avalon shifts the test ground:**

- No closed-form optimal policy. Play is **heuristic deduction + social inference**, not symbolic computation.
- Naive-bot baselines exist (AvalonBench publishes them).
- A single LLM under test against rule-based bots avoids multi-LLM confounds.
- It is the explicit generalisation case in [`proposal.md`](proposal.md) Task 13.

If `B` is a real difficulty knob, Avalon is where it should appear cleanly.

---

## 2. Reference: AvalonBench

- **Paper:** Light et al., "AvalonBench," NeurIPS 2023 GamesAndAI workshop.
- **Repo:** [github.com/jonathanmli/Avalon-LLM](https://github.com/jonathanmli/Avalon-LLM) (built on AgentBench).
- **Config we will use:** `avalon-dev-single` — **one** LLM under test against **four** rule-based naive bots. Single-LLM-vs-bots avoids multi-agent reasoning confounds entirely.
- **Roles in Avalon:** {Merlin, Percival, Loyal Servant} are good-side; {Morgana, Assassin, Minion} are evil-side. Each has different information sets and incentives — Merlin sees evil except Mordred; Percival sees Merlin and Morgana indistinguishably; etc.

The repo provides:

- Game env (state machine, mission-vote-quest loop).
- Naive-bot opponents (rule-based, deterministic-ish).
- Prompt templates for each role.
- A reference LLM agent that talks to the env.

We extend their published harness rather than build from scratch (per the proposal's stated approach).

---

## 3. Integration approach

The cleanest path is to **vendor AvalonBench under `third_party/`** (or add as a git submodule) and wrap their env + agent loop with a thin shim that logs into our existing tracking infrastructure.

### 3.1 What needs to be built

| Component | Effort | Notes |
|-----------|--------|-------|
| `third_party/avalon_llm/` | trivial | Submodule or subtree-import. |
| `src/cot_knob/games/avalon.py` | small | A `GameAdapter` class that wraps AvalonBench's env to look like our `GameState` protocol (or, alternatively, a fully separate `AvalonRunner` that bypasses our `runner.play_match`). |
| `src/cot_knob/agents/avalon_llm_agent.py` | small | An `Agent` that invokes our `LLMClient` against AvalonBench's prompt format. Reuses our `OllamaClient` / `SGLangClient`. |
| Logging shim | small | After each AvalonBench game, write a `trial_end` event to JSONL + a row to `trials` in `results.db`. Per-action logging optional in v1. |
| `LLMClient` parity check | small | Validate `max_tokens`, `temperature`, system-prompt handling match what AvalonBench expects. Same lesson as the R1-vs-Llama Nim pivot. |
| Sweep config plumbing | small | Add `game: avalon` and an Avalon-specific sub-config (role, opponent profile) to `SweepConfig`. |

Total: **~1–2 days of integration work** before any Avalon experiment can run.

### 3.2 What we deliberately reuse

- `Store` (SQLite) and `JSONLWriter` — write Avalon trials into the same schema. Aggregate analysis stays unified across games.
- `LLMClient` interface — `OllamaClient` and (eventually) `SGLangClient` work unchanged.
- Run-name → directory convention (`run_name__shortid`).
- Wilson CI / exact binomial test analysis pipeline.

---

## 4. Anchors (Nim-style methodological hygiene)

The Nim pilot taught us that the wrong baseline produces unfalsifiable headlines. Avalon needs analogous anchors:

| Anchor | What it is | Source |
|--------|-----------|--------|
| **Naive-bot vs naive-bot win rate** | Good-side win % when all five players are AvalonBench's rule-based bots. | The AvalonBench paper reports this empirically with N=1000+. Re-verify with our shim before the headline result. |
| **B=0 LLM (random-action) anchor** | LLM-under-test plays only random/default actions because reasoning is truncated. | Run as one cell of E1 (below). Should match the naive-bot anchor within sampling noise (analogous to Nim B=0 = 53% empirical / 50% theoretical). |
| **Prompt-only baseline** | LLM is asked at low budget with no scaffolding — measures pattern-matching contribution alone. | Optional confirmatory anchor, run as part of E2. |

Unlike Nim, Avalon does **not** have a closed-form theoretical anchor. The naive-vs-naive empirical anchor at large N replaces it.

---

## 5. Experiment matrix

### E1 — Headline budget sweep (PRIMARY)

**Question:** Does CoT budget produce a non-flat `W(B)` curve when the LLM plays a fixed Avalon role against naive bots?

| Variable | Value |
|----------|-------|
| Role under test | **Merlin** (good-side, hardest reasoning load — must hide identity while guiding good team) |
| Opponent | 4 naive bots (default AvalonBench config) |
| Budget B | {64, 256, 1024} (drop 64 if it is parse-fail dominated, like Nim) |
| Prompt | Minimal scaffolding (see E2 for variants) |
| N | **30 per cell** |
| Memory | Full chronological history (default) |
| Decoding | T=0 (greedy) |

**Total games:** 90.

**Anchors plotted alongside:** naive-bot vs naive-bot win rate (large N from AvalonBench paper); B=0 LLM anchor.

**What "positive" looks like:**

- Win rate at B=1024 ≥ naive-bot anchor + ~15 pp, with non-overlapping CIs.
- Monotone-ish curve (B=64 ≤ B=256 ≤ B=1024 within sampling noise).

**What "negative" locks in:**

- Win rate at all budgets within ±10 pp of naive-bot anchor. → "Llama 8B does not exceed naive-bot heuristic play on Avalon-Merlin even at B=1024." That is itself a clean negative result and arguably a more interesting headline than the proposal's "monotone curve" pre-registration.

### E2 — Prompt robustness matrix (CO-PRIMARY)

**Question:** Does the budget effect (if any) survive prompt redesign? This directly addresses the prompt-confound risk you flagged earlier — that the curve might really be `W(B | prompt)`.

| Variable | Value |
|----------|-------|
| Prompt regimes | {minimal, structured-stage, role-conditioned} |
| Budget B | {64, 1024} (endpoints only) |
| Role | Merlin (matches E1) |
| N | **20 per cell** |

**Cells:** 3 prompts × 2 budgets = 6.
**Total games:** 120.

**Prompt regimes:**

- **Minimal:** rules + state + "What's your action? Reasoning:". No scaffold. Same prompt across B.
- **Structured-stage:** "First, list what you know about each player. Second, identify suspicious behavior. Third, decide your action." Same across B.
- **Role-conditioned:** Merlin-specific: "You see evil except Mordred. Do not reveal this. First, recall the evil players you know. Second, …"

**What "positive" looks like:**

- Curve direction (B=64 → B=1024) is consistent across prompts (≥2 of 3 prompts show same sign).
- Best-prompt × best-budget significantly beats worst-prompt × B=64 — but the **budget-within-prompt** trend is stable.

**What "negative" locks in:**

- Sign of (B=1024 − B=64) flips across prompts → "budget effect is prompt-dependent; reporting a single curve would have been a single-prompt artifact, not a budget effect."

### E3 — Memory ablation

**Question:** Does the choice of memory representation (full history vs structured summary) confound the budget effect on Avalon? This was Risk 5 in the original proposal and is sharper on Avalon than on Reversi because dialogue context blows up by mission 3.

| Variable | Value |
|----------|-------|
| Memory | {full chronological history, capped structured summary} |
| Budget B | 512 (mid-range; B=1024 if 512 is parse-failed too often) |
| Role | Merlin |
| Prompt | Minimal (matches E1) |
| N | **30 per cell** |

**Cells:** 2.
**Total games:** 60.

**Auxiliary measurement:** for the structured-summary condition, log the summary text on every turn. Compute BLEU + embedding similarity between summaries produced for the same game state at different budgets (proposal Task 5 sub-analysis). Tests whether the summariser is itself budget-correlated.

**What "positive" looks like:** memory choice does not change `W` by more than ~10 pp at the same budget; summary stability is high. → "Memory is not a hidden confound."

**What "negative" looks like:** large win-rate gap by memory mode; summary similarity drops with B. → "Reported budget effect is partly a summariser-quality effect; need to control for it before publishing."

### E4 — Role coverage (lighter)

**Question:** Does CoT budget matter more for evil roles (must lie strategically) than for good roles (must deduce)?

| Variable | Value |
|----------|-------|
| Roles | {Merlin, Percival, Loyal Servant, Morgana, Assassin, Minion} |
| Budget B | {64, 1024} (endpoints only) |
| Prompt | Minimal |
| N | **15 per cell** (lighter than E1) |

**Cells:** 6 roles × 2 budgets = 12.
**Total games:** 180.

**Note:** Loyal Servant and Minion are the simplest information sets (no special knowledge). Run them as the "control" pair if budget time is tight.

**What "positive" looks like:** the size of `W(1024) − W(64)` is consistently larger for evil roles than good roles (or vice versa). → "Reasoning load interacts with role type."

**What "negative" looks like:** flat across all roles, or noisy direction. → "Budget effect (if any) is role-independent on Avalon."

---

## 6. Outcome metrics

For every cell, report:

1. **Good-side win rate** (Avalon's headline outcome). Wilson 95% CIs.
2. **Per-mission action quality**, if AvalonBench exposes phase labels. Stratify by mission round (proposal Task 12 style).
3. **Action-failure rate** — how often the LLM produces an unparseable / illegal action and the harness has to default. Direct analogue of Nim `parse_failed`. Critical for distinguishing "model failed to reason" from "model lost the game."
4. **Memory summary stability** (E3 only) — BLEU + embedding similarity of summaries across budgets.
5. **Reasoning trace** — the full `pass1_text` per turn, stored in JSONL. Enables post-hoc failure-mode taxonomy analogous to Nim §9.

---

## 7. Open questions / blockers

These need to be answered before E1 can run:

- **Does AvalonBench's prompt loop expose a token budget knob?** If their wrapper hard-codes `max_tokens`, we have to patch their agent class to take `B` as a parameter.
- **How is action selection actually made?** If they use a Pass-2-style constrained selection from a list of legal actions, we are back in the Reversi-OOD territory and need to verify it isn't a load-bearing scaffold (cf. [`ood_probe_findings.md`](ood_probe_findings.md) v1 vs v2).
- **Deterministic seeding inside their env?** We need reproducible role assignment + naive-bot behaviour for paired comparisons.
- **Multi-turn dialogue truncation policy?** Avalon's history grows mission by mission; what happens when it hits the model's context length? AvalonBench likely already has a default; we need to know what it is and whether it varies by `B`.

These are answered by the integration spike (build `avalon.py` + run one game end-to-end before any sweep).

---

## 8. Cost estimates

Avalon games are longer than Nim games. Rough envelope:

| Item | Estimate |
|------|----------|
| Turns per game | ~30–60 (5 missions × ~6–12 player turns) |
| LLM calls per game | ~10–20 (depending on whether discussion phases are voiced) |
| Tokens per call (B=1024) | ~1024 reasoning + few-hundred system + few-hundred history |
| Single game wallclock @ 200 tok/s (5080 single-stream est.) | ~1–2 min |
| **E1 (90 games)** | ~2–3 hr |
| **E2 (120 games)** | ~3–4 hr |
| **E3 (60 games)** | ~1–2 hr |
| **E4 (180 games)** | ~4–6 hr |
| **Total** | ~10–15 hr at single-stream Ollama; ~3–5 hr at SGLang batched. |

**Calibrate first:** Run one full Avalon game end-to-end and measure tokens-emitted / wallclock. Update this table on first contact with the env.

---

## 9. Order of execution

1. **Spike (1–2 days):** vendor AvalonBench, write `avalon.py` adapter, run **one** end-to-end game with Ollama Llama 8B at B=1024 from a fixed seed. Verify telemetry flows to JSONL + SQLite. **Stop and write up what you learned about the env's API.**
2. **Anchor verification (~30 games):** naive-bot vs naive-bot at large N if AvalonBench's published number is older than 2024, otherwise cite their value.
3. **E1 (90 games):** the headline budget sweep on Merlin.
4. **STOP-AND-REASSESS POINT.** If E1 is flat against naive-bot anchor (analogous to Nim's flat curve), that is a publishable negative result combined with the Nim story. Decision: extend Avalon (E2/E3) or pivot to a different game.
5. **E2 (120 games):** prompt-robustness matrix. Definitive on whether the curve is prompt-dependent.
6. **E3 (60 games):** memory ablation. Run only if E1 and E2 are positive — otherwise the memory confound doesn't matter for an already-flat curve.
7. **E4 (180 games):** role coverage. Run last; least likely to change the headline.

If GPU time is tight: **the minimum viable Avalon contribution is the spike + E1.** Everything else is hardening.

---

## 10. Lessons from Nim baked into this plan

| Nim lesson | How it shapes the Avalon plan |
|------------|--------------------------------|
| Wrong anchor → unfalsifiable headline. | Use naive-bot vs naive-bot as the explicit comparator; don't infer it from B=0. |
| N=10 is noise; N=30 is suggestive; N=85 is conclusive. | E1 / E3 use N=30 minimum; E2 / E4 are deliberately exploratory at lower N. |
| Counterbalanced random play is a fair coin regardless of structural advantage. | For Avalon, role assignment is asymmetric — fix role per cell rather than counterbalance. |
| Effective budget < requested budget if parse fails dominate. | Drop B=64 from primary plots if action-failure rate is ≥ 50% there. |
| Trace inspection finds failure modes that aggregate metrics hide. | After E1, do a Nim §9-style failure-mode taxonomy on 5–10 representative traces before declaring a result. |
| Different games stress different model capabilities. | Don't assume Avalon results transfer back to Nim or Reversi without re-running. |

---

## 11. What this doc deliberately does not cover

- **Multi-LLM Avalon** (multiple LLMs at the table). AvalonBench supports it; we are explicitly not using it (proposal Task 13 = single LLM vs bots = isolates one variable).
- **The adaptive controller** (proposal Task 10). Predicated on a non-flat `W(B)` curve. Revisit after E1 + E2.
- **Reversi as the primary game.** Reversi is the proposal target; Avalon is the generalisation case. Reversi work resumes only after a usable model + harness is identified.

---

## 12. Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | Plan drafted with E1 (Merlin) as gating headline; E4 (multi-role) deferred. | E1 is the smallest experiment that can either confirm or kill the budget-as-knob hypothesis on Avalon. Multi-role is interesting but only after we know the headline holds. |
