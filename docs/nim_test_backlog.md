# Nim Test Backlog — High-N Follow-Up Experiments

**Audience:** Self / future me on RTX 5080 Blackwell.
**Status:** Forward-looking. None of these have been run yet.
**Companion docs:** [`nim_experiment_findings.md`](nim_experiment_findings.md) (current results), [`pc_dev_setup.md`](pc_dev_setup.md) (how to bring the rig up).

---

## 1. State of play (May 2026)

The N=30 Nim sweeps and diagnostics are complete on Mac (Llama 3.1 8B Instruct via Ollama). The headline:

| Cell | Win rate (N=30) | 95% Wilson CI | p vs 50% |
|------|----------------|----------------|----------|
| `free_cot` B=1024 | 40.0% | [25%, 58%] | 0.36 |
| `nim_sum_given` B=1024 | 50.0% | [33%, 67%] | 1.00 |
| `few_shot` B=1024 | 53.3% | [36%, 70%] | 0.86 |
| `step_by_step` B=1024 | 53.3% | [36%, 70%] | 0.86 |
| `step_by_step` B=256 | 43.3% | [27%, 61%] | 0.58 |
| B=0 empirical (no LLM) | 53.3% | [36%, 70%] | 0.86 |
| **Theoretical anchor** | **50.0%** | exact | — |

**What is conclusive at N=30:**

- Llama 8B on Nim [3,5,7] is **at chance level** for every prompt variant and budget tested.
- The model **does not condition on nim-sum structure** — [1,2,3] forced-loss matches [3,5,7] baseline.
- The earlier "~80% structural prior" was N=10 noise; the correct anchor is 50.0% by exact game-tree recursion.

**What is suggestive but not conclusive:**

- Structured prompting (`step_by_step` / `few_shot` / `nim_sum_given`) sits ~13pp above unstructured `free_cot` (53% vs 40%). N=30 cannot resolve a 13pp gap at p<0.05.
- Within `step_by_step`, B=256 → B=1024 trends 43% → 53%. ~10pp. Same statistical story.

**What is unknown:**

- Does the structured-vs-unstructured gap survive at N≥85?
- Does the budget gradient inside `step_by_step` survive at N≥85?
- Does a harder Nim configuration (more turns, wider-bit XOR) widen the skill–chance gap, or does Llama 8B simply fail at any Nim?
- Does temperature > 0 collapse / amplify the curve shape?

These are the questions the backlog below is designed to resolve.

---

## 2. Power calculation reminder

| N per cell | 95% CI half-width (at p=0.5) | Min detectable difference at 80% power |
|------------|------------------------------|-----------------------------------------|
| 30  | ± 18 pp | ~33 pp |
| 50  | ± 14 pp | ~25 pp |
| 85  | ± 11 pp | **~20 pp** |
| 150 | ± 8 pp  | ~15 pp |
| 300 | ± 6 pp  | ~10 pp |

Targeted effect sizes:

- **~13 pp** (structured vs unstructured): needs N ≈ 200/cell.
- **~10 pp** (budget within step_by_step): needs N ≈ 300/cell.
- **~20 pp**: comfortably resolvable at N ≈ 85/cell.

The backlog uses N=85 as the "definitive negative" cutoff: if a 20pp lift doesn't show by then, the headline finding ("Llama 8B is at chance on Nim regardless of prompt") is locked in. Tighter resolution (10–13 pp) is reserved for the few cells where a positive trend is already directional.

---

## 3. Test backlog

| ID | Priority | Hypothesis tested | Config | N/cell | Cells | Total games |
|----|----------|-------------------|--------|--------|-------|-------------|
| **T1** | High | `step_by_step` budget gradient (43% → 53%) is real | [3,5,7], `step_by_step`, B ∈ {256, 512, 1024}, vs random | 85 | 3 | 255 |
| **T2** | High | Structured (`step_by_step`) > unstructured (`free_cot`) by ~13 pp at B=1024 | [3,5,7], B=1024, vs random | 85 | 2 | 170 |
| **T3** | Medium | Harder Nim ([7,11,13]) widens skill–chance gap | [7,11,13], B ∈ {256, 512, 1024, 2048}, vs random + vs optimal | 30 | 8 | 240 |
| **T4** | Medium | [1,2,3] forced-loss mirror confirms model ignores nim-sum at N=85 | [1,2,3], `{free_cot, step_by_step}` × B=1024, vs random | 85 | 2 | 170 |
| **T5** | Low | Curve shape survives non-greedy decoding | [3,5,7], `free_cot`, T=0.7, B ∈ {256, 1024} | 30 | 2 | 60 |
| **T6** | Low | parse_failed splits cleanly by `nim_sum at LLM turn` | [3,5,7], vs optimal, B ∈ {256, 512, 1024}, instrumented logging | 30 | 3 | 90 |

Total games if all run: **~985** trials. Realistically T1+T2 (~425 games) is the minimum value set; T3 is the most interesting *secondary* test if T1/T2 turn up a positive effect.

---

## 4. Detailed test specs

### T1 — `step_by_step` budget gradient at high N (HIGH PRIORITY)

**Question:** Inside the structured-prompt regime, does CoT budget actually do anything?

**Setup:**

```yaml
run_name: nim_t1_step_budget_n85
game: nim
nim_piles: [3, 5, 7]
budgets: [256, 512, 1024]
n_per_cell: 85
llm_plays: both
prompt_variant: step_by_step
opponent: { kind: random }
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.0 }
```

**What "positive" looks like:**

- B=256 ≤ B=512 ≤ B=1024 monotone, with B=1024 vs B=256 gap ≥ 15 pp (resolvable at p<0.05 with N=85).
- B=1024 win rate significantly > 50% (binomial p < 0.05 against the theoretical anchor).

**What "negative" locks in:**

- All three cells within ±10 pp of 50%, all binomial p > 0.05. → "Even with a procedure given, budget does not produce above-chance play on [3,5,7]."

**Estimated wallclock (5080, placeholder):** TBD — calibrate with one B=1024 game on the rig. Mac throughput is ~60–80 tok/s; expect 3–5× speedup on Blackwell single-stream, more with batching.

---

### T2 — Variant lock-in at high N (HIGH PRIORITY)

**Question:** Does the ~13pp `step_by_step` lift over `free_cot` at B=1024 survive replication?

**Setup:**

```yaml
run_name: nim_t2_variant_lockin_n85
game: nim
nim_piles: [3, 5, 7]
budgets: [1024]
n_per_cell: 85
llm_plays: both
opponent: { kind: random }
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.0 }
# Run twice with different prompt_variant: free_cot, step_by_step
```

**What "positive" looks like:**

- `step_by_step` − `free_cot` ≥ 15 pp with non-overlapping CIs, both significantly distinguishable (p < 0.05) by exact 2×2 Fisher test on win counts.

**What "negative" locks in:**

- Gap ≤ 10 pp or CIs overlap. → "Prompt structure does not differentiate at N=85; the headline finding holds."

---

### T3 — Larger-game sweep on Nim [7,11,13] (MEDIUM PRIORITY)

**Question:** Does a harder Nim config — more turns, wider-bit XOR — produce a non-flat budget curve, or is Llama 8B simply incapable on Nim of any size?

**Pre-step (mandatory):** Compute the exact per-config random-vs-random anchor for [7,11,13]. The [3,5,7] anchor of 50% does **not** transfer; this must be recomputed by exact recursion (or N=10⁵ Monte Carlo if the state space is too large for full enumeration).

**Setup:**

```yaml
run_name: nim_t3_larger_game
game: nim
nim_piles: [7, 11, 13]   # nim_sum = 9, P1 wins under optimal play
budgets: [256, 512, 1024, 2048]
n_per_cell: 30
llm_plays: both
prompt_variant: free_cot          # also run step_by_step as a follow-up
opponent: { kind: random }        # also run vs optimal
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.0 }
```

**What "positive" looks like:**

- vs random: a budget cell exceeds the (recomputed) random anchor by ≥ 20 pp, p < 0.05.
- vs optimal: any cell wins > 0% (Nim [3,5,7] showed 0% across the board; > 0 here would mean the model occasionally finds the optimal response from a forced winning position).

**What "negative" locks in:**

- All cells within sampling noise of the random anchor; vs-optimal still 0%. → "The Nim chance ceiling is not a function of game size; it's a function of XOR execution reliability."

**Methodological reminder:** Always report the per-config random anchor with the curve, not the [3,5,7] 50% number.

---

### T4 — Forced-loss [1,2,3] mirror at high N (MEDIUM PRIORITY)

**Question:** Does the [1,2,3]-vs-[3,5,7] equivalence finding (model ignores nim-sum) survive at N=85?

**Setup:**

```yaml
run_name: nim_t4_forced_loss_n85
game: nim
nim_piles: [1, 2, 3]    # nim_sum = 0, P1 forced to lose with optimal play
budgets: [1024]
n_per_cell: 85
llm_plays: both
opponent: { kind: random }
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.0 }
# Two runs: prompt_variant in {free_cot, step_by_step}
```

**What "positive" looks like (model uses nim-sum):**

- LLM-as-P1 win rate drops sharply on [1,2,3] vs the corresponding [3,5,7] cell.
- LLM-as-P2 win rate rises above the [3,5,7] P2 baseline.

**What "negative" locks in:**

- P1/P2 split on [1,2,3] ≈ split on [3,5,7] within ±10 pp. → "The model is not conditioning on nim-sum; observed play is nim-sum-blind, by trait."

---

### T5 — Stochastic-decoding robustness (LOW PRIORITY)

**Question:** Does the at-chance result survive non-greedy sampling, or is it a property of the deterministic decode path?

**Setup:**

```yaml
run_name: nim_t5_stochastic
game: nim
nim_piles: [3, 5, 7]
budgets: [256, 1024]
n_per_cell: 30
llm_plays: both
prompt_variant: free_cot
opponent: { kind: random }
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.7 }
```

**What "positive" looks like:**

- Win rate at T=0.7 differs from T=0 by ≥ 20 pp (either direction). Suggests the deterministic decode is forcing the model into a single failure mode, and sampling lets it occasionally stumble onto correct play (or vice versa).

**What "negative" locks in:**

- Same ~50% chance result at T=0.7. → "Failure mode is robust to decoding randomness."

---

### T6 — Position-stratified parse_failed (LOW PRIORITY)

**Question:** Does the 5.2 finding (parse_failed elevated for LLM-as-P2 vs optimal at B≥256) generalize across budgets and split cleanly by `nim_sum at LLM turn`?

**Setup:** Re-run vs-optimal sweep with one new instrumentation: log `nim_sum_at_llm_turn` for every parse_failed turn. Stratify aggregate parse_failed by this label.

```yaml
run_name: nim_t6_pos_strat_parsefail
game: nim
nim_piles: [3, 5, 7]
budgets: [256, 512, 1024]
n_per_cell: 30
llm_plays: both
prompt_variant: free_cot
opponent: { kind: optimal }
model: { backend: ollama, name: llama3.1:8b, think: false, temperature: 0.0 }
```

**What "positive" looks like:**

- parse_failed in losing positions (`nim_sum = 0` at LLM turn) systematically higher than in winning positions, replicating §5.2 across budgets.

**What "negative" locks in:**

- Flat parse_failed by position type. → "§5.2 was an artifact of the small N=10 sample, not a robust position-type effect."

---

## 5. Order of execution

If GPU time is constrained, run in this order:

1. **T1 (255 games)** — directly tests the most concrete suggestive finding.
2. **T2 (170 games)** — locks in or rejects the structured-vs-unstructured headline.
3. **STOP-AND-REASSESS POINT.** If both T1 and T2 are negative, the Nim-pilot story is "Llama 8B is robustly at chance on Nim [3,5,7] across all prompts and budgets we tested." Move directly to Avalon (see [`avalon_test_plan.md`](avalon_test_plan.md)).
4. **T3 (240 games)** — only worth running if T1 or T2 turn up a positive effect, or if you want a clean negative on a *harder* Nim configuration before declaring "Nim ceiling is real."
5. **T4 (170 games)** — locks in the "ignores nim-sum" finding more rigorously. Lower priority because the N=10 result was already directionally clear.
6. **T5 (60 games)** — confirmatory. Quick.
7. **T6 (90 games)** — confirmatory. Quick.

---

## 6. Cost estimates (placeholder — calibrate on rig)

| Throughput | Avg game length (B=1024) | Single-game wallclock | T1+T2 wallclock | All tests wallclock |
|-----------|--------------------------|-----------------------|------------------|---------------------|
| 60 tok/s (Mac single-stream) | ~10 turns × 1024 tok = 10 240 tok | ~170 s | ~20 hr | ~46 hr |
| 200 tok/s (5080 single-stream, est.) | same | ~50 s | ~6 hr | ~14 hr |
| 600 tok/s (5080 batched 4-way, est.) | same | ~17 s | ~2 hr | ~5 hr |

**Action item:** First task on the new box is to measure a single-stream Llama 8B Q4_K_M tok/s with Ollama and update this table. See [`pc_dev_setup.md`](pc_dev_setup.md) §smoke test.

---

## 7. Cross-cutting hygiene

For every test run:

- **Always recompute and report the random-vs-random anchor for the specific pile config.** Do not transfer the [3,5,7] 50% number to other configs without a re-derivation.
- **Counterbalance:** `llm_plays: both` (50/50 P1/P2) for every cell.
- **Use exact binomial tests** vs the per-config anchor for the headline statistical claim. Wilson CIs for plotting.
- **Save the prompt** (already wired into JSONL via `pass1_prompt` / `pass1_system`).
- **Use human-readable run-dir names** (`run_name__shortid`) — already wired.
- **For T3 (config change), update notebooks 06/07** to load the new run dirs and recompute the anchor cell.

---

## 8. What this backlog deliberately does not include

- **Reversi sweeps.** Reversi is the proposal target; the OOD probe ([`ood_probe_findings.md`](ood_probe_findings.md)) shows R1-Distill-7B can't derive Reversi legality without scaffolding. Reversi work is gated on either (a) a model that *can* derive legality (Llama 8B was not tried; worth a quick OOD repro) or (b) accepting the Pass-2 legal-list scaffold and reframing the headline as "strategic discrimination given a pre-validated legal set."
- **Adaptive controller (proposal Task 10).** Predicated on a usable static `W(B)` curve. The Nim curve is flat; nothing to control. Revisit after Avalon.
- **Filler-arithmetic confound (proposal Task 4).** Same — predicated on a non-flat `W(B)`.

---

## 9. Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | Backlog drafted with T1/T2 as gating tests | N=30 results were suggestive but underpowered; one targeted high-N pass either locks in or rejects the headline before further variant work. |
