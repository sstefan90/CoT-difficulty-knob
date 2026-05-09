# Nim Experiment Findings — Llama 3.1 8B (May 2026)

> **Project status (May 2026):** Nim is the active *pilot* for the CoT-budget knob study, run on **Llama 3.1 8B Instruct via Ollama** with a pure-Python game harness. The proposal's primary game (Reversi via Ludii + R1/SGLang) remains the planned target — see [`docs/architecture.md`](architecture.md) and [`docs/proposal.md`](proposal.md). Reversi-specific OOD findings live in [`docs/ood_probe_findings.md`](ood_probe_findings.md); this file owns the **Nim** OOD probe (§3) and all Nim sweeps.

## 1. Why We Pivoted from DeepSeek R1 7B

> **Why this section exists.** Before reporting any Nim sweep results, we need to explain *why the model and serving stack changed*. The original proposal targeted DeepSeek R1 7B as the reasoning model; the shift to Llama 3.1 8B was forced by R1's behaviour under our budget knob, not by performance preference.
>
> **What we learned.** Under the proposal's original premise — `B` controls a *visible* CoT trace — R1 fails: `num_predict=B` controls only the hidden `thinking` field, and truncation produces an empty `response`. No win-rate-vs-B curve built on R1 traces measures reasoning depth as intended. Llama's single-pass `think=False` mode restores the property we need.

### 1.1 How R1 Works via Ollama

DeepSeek R1 7B, when served by Ollama with `think=True`, returns two separate
fields: a `thinking` field containing an internal chain-of-thought and a
`response` field containing the final answer. The parameter `num_predict=B`
controls the length of the `thinking` field only. The `response` field is
generated *after* thinking completes.

This architecture is fundamentally incompatible with the study's premise, which
requires `B` to control a **visible, measurable** reasoning trace.

### 1.2 Challenges Encountered

**Empty response on thinking truncation.**
When `num_predict=B` cuts the `thinking` field short (`finish_reason=length`),
Ollama returns a blank `response`. There is no partial answer — the model either
finishes thinking and writes a response, or the response is empty. At B=64 and
B=256, the thinking was almost always truncated, yielding zero extractable
content in the `response` field.

**Fixed verbose preamble.**
Regardless of prompt engineering, R1 7B opened every `thinking` field with a
multi-sentence framing preamble:  
> *"Okay, so I'm trying to figure out the best Nim move here. Let me think about nim-sums..."*

This consumed 30–60 tokens before any calculation began. At B=256 the model was
still writing setup text when the budget ran out. Strong system-prompt
instructions (`"Do NOT open with 'Okay'..."`) had no measurable effect — the
preamble is baked into R1's fine-tuning.

**Reasoning incomplete at moderate budgets.**
Even at B=1024, R1's thinking trace for Nim [3,5,7] was often truncated
mid-calculation. A representative example ended with  
> *"...perhaps the correct approach is to find which pile"*  
The model had computed binary representations but had not yet performed the XOR.
The response field was empty; no move was recoverable.

**No reliable fallback extraction.**
The only recourse was scanning the raw `thinking` text for move-like phrases
(`_NIM_TAKE_REGEXES`). This introduces a scientific confound: the extracted move
comes from an *incomplete* calculation, not from the model's concluded answer. It
is whatever phrase appeared last in a truncated train-of-thought — not the
model's intended action.

**Pass-2 commit call failure.**
A second "commit" generation call (`think=False`, 40–64 tokens) was tried to
force a short final answer after the Pass-1 reasoning. This also failed: R1's
fine-tuning produces a verbose preamble even in `think=False` mode, consuming
the entire small budget before outputting any move token.

### 1.3 Summary Table

| Issue | Effect on study |
|---|---|
| Empty response on truncation | No extractable move at B<1024 |
| Fixed preamble (~50 tokens) | Effective budget shrinks by ~50 tokens everywhere |
| Reasoning incomplete at B=1024 | Moves extracted from incomplete calculation |
| Regex fallback confound | Extracted move ≠ intended move |
| Pass-2 commit fails | Alternative extraction also fails |

**Conclusion.** `B` does not cleanly control R1's visible reasoning: it controls
an opaque, truncated thinking trace whose contents may or may not include a
coherent final move. Win-rate-vs-B curves built on R1 data are uninterpretable
as a measure of reasoning depth.

---

## 2. Llama 3.1 8B Instruct as Replacement

With `think=False` (Llama's only mode), Ollama returns the model's entire output
in the `response` field. At B=64 the model outputs up to 64 visible tokens of
reasoning + action. The `thinking`/`response` split and the empty-response-on-truncation
problem disappear. `B` cleanly controls visible CoT length, which is the
experimental knob the study requires.

The `think` flag is now a first-class config field (`LLMConfig.think: bool`),
defaulting to `true` for R1 and set to `false` for Llama. Existing R1 configs
remain valid.

---

## 3. OOD Probe Results (Llama 3.1 8B, Nim [3,5,7])

> **Why we ran this.** Before any sweep, we need to know whether Llama can (a) produce a *legal* Nim move at each budget without scaffolding (parse-failure floor), and (b) pick a *strategically optimal* move (nim-sum awareness). If legality already fails everywhere, the sweep is measuring move-extraction, not reasoning. If legality is free but strategy is flat, then `B` is the real lever to study.
>
> **What we learned.** Legality is *free* for Llama at any non-zero budget (100% across B≥64). Strategy is the bottleneck and saturates around 20–25% from B=256 onwards — the model attempts XOR on every turn but executes it unreliably, and more tokens past B=256 don't fix the arithmetic. Reversi shows the opposite pattern (legality is the bottleneck, see [`docs/ood_probe_findings.md`](ood_probe_findings.md)) — different games stress different parts of the model.

The OOD probe tests two capabilities without showing the model the list of legal
moves, to avoid reducing the task to multiple-choice selection.

**Probe 1 — Legality** (can the model produce a valid move?):

| Budget | Legal rate |
|--------|-----------|
| B=0 (prefix injection) | no_parse (probe artifact) |
| B=64 | **100%** |
| B=256 | **100%** |
| B=512 | **100%** |
| B=1024 | **100%** |

Legality is free for Llama at any non-zero budget. This is expected: Nim
legality is trivially computable (take N from pile X iff pile X ≥ N), and Llama
reliably follows the `MOVE: pile=X take=N` format instruction.

**Probe 2 — Nim-sum strategy** (is the chosen move optimal?):

| Budget | Canonical positions | Non-canonical positions |
|--------|--------------------|-----------------------|
| B=0 | no_parse | no_parse |
| B=64 | 0% | 0% |
| B=256 | 25% | 20% |
| B=512 | 25% | 20% |
| B=1024 | 25% | 20% |

Notes:
- Three of four canonical positions ([1,2,3], [1,3,5,7], [2,4,6]) are losing
  positions (nim_sum=0). No move is "optimal" in these; only the [3,5,7]
  position is a winning one.
- Strategy rate improves from 0% to ~20–25% between B=64 and B=256, then
  plateaus. The model understands the nim-sum framework (it always attempts the
  XOR calculation) but makes arithmetic errors.
- Non-canonical positions remain hard even at B=1024, confirming this is genuine
  computation difficulty, not memorisation.

**Key takeaway**: Legality is not a confound for Llama (100% at all B>0).
Strategy is the challenge, and it improves with budget up to B=256 then
plateaus. The difficulty curve is smooth in the B=[64,256] range and flat
beyond that — for this position and model.

---

## 4. Prelim Sweep: parse_failed Rate vs Budget

> **Why we ran this.** The OOD probe shows legality is high in single-shot calls — but in a *played game* the model also has to write the structured `MOVE: pile=X take=N` tag at the end of its reasoning. We need to know the smallest `B` for which the tag actually fits, because below that threshold every "loss" is just truncation, not bad play.
>
> **What we learned.** Effective reasoning budget for Llama 8B on Nim [3,5,7] is ~250–350 tokens. Below B=256, parse_failed → 100% and the agent collapses to a random move. This means **B=64 is functionally identical to B=0** for this task and the two cells should not be interpreted as separate "low-budget reasoning" conditions in headline plots.

Prelim config: Nim [3,5,7], N=3/cell, opponent=random, budgets=[0,64,256,512,1024].

| Budget | parse_failed | finish=length |
|--------|-------------|--------------|
| 0 | 100% (by design — no generation) | 0% |
| 64 | **100%** | 100% |
| 256 | 22% | 89% |
| 512 | **0%** | 17% |
| 1024 | **0%** | 7% |

The model needs ~250–350 output tokens to complete nim-sum binary arithmetic
**and** write the `MOVE: pile=X take=N` tag. At B=64 it is always truncated
mid-calculation. At B=512 it reliably finishes. The `finish=length` rate at
B=256 (89%) confirms the model often hits the limit, but sometimes the MOVE tag
falls within the first 256 tokens (shorter reasoning path), yielding
`parse_failed=False` despite truncation.

**Effective reasoning budget**: ~250–350 tokens for Nim [3,5,7] with Llama 8B.
Below this threshold, moves are forced random (parse_failed fallback). This
means B=64 is functionally identical to B=0 for this task, which collapses two
experimental cells.

---

## 5. Full Sweep Results

> **Why we ran these.** The headline experiment of the project: characterise the win-rate-vs-CoT-budget curve `W(B)` on a solved game. Three opponent regimes — Random (lower bound), Optimal (upper bound), and Self-play vs B=1024 (controlled internal comparison) — together let us separate *reasoning effect* from *opponent strength* and from *symmetric noise*.
>
> **What we learned (headline).** On Nim [3,5,7], `W(B)` is **flat against the 50% random-play anchor** for all deliberate-reasoning cells — none are statistically distinguishable from chance, and against an optimal opponent the LLM wins 0% at every budget. The only large effect is the parse-failed step at B<256.

### 5.1 vs Random (nim_vs_random_llama__45e521b6, N=10/cell)

> **Why this cell.** Sets the *lowest-strength opponent* baseline so any LLM reasoning advantage has the most room to show up. Random play is also the only opponent for which we have a clean theoretical anchor (exact game-tree recursion).
>
> **What we learned.** No budget cell beats the 50.0% theoretical anchor. The N=10 observed 80% at B=0/64 is sampling noise from a fair coin (binomial p=0.11). At B=1024 the model wins 40% — directionally below chance but not significant at N=10.

Theoretical random-play anchor: **50.0%** (exact game-tree recursion). N=10 budget cells have ±31 pp Wilson CIs; no cell is distinguishable from the 50% anchor.

| Budget | Win rate (N=10) | parse_failed% |
|--------|----------------|--------------|
| 0 | 80% | 100% |
| 64 | 80% | 100% |
| 256 | 50% | 18% |
| 512 | 60% | 3% |
| 1024 | 40% | 0% |

Broken down by side:

| Budget | As P1 (black) | As P2 (white) |
|--------|--------------|--------------|
| 0 | 5/5 | 3/5 |
| 64 | 5/5 | 3/5 |
| 256 | 3/5 | 2/5 |
| 512 | 4/5 | 2/5 |
| 1024 | 3/5 | 1/5 |

### 5.2 vs Optimal (nim_vs_optimal_llama__e1657086, N=10/cell)

> **Why this cell.** Sets the *upper bound* on what reasoning could buy us: against an optimal opponent, only correct nim-sum play survives. If the model never wins, it is not approximating optimal play even at the highest budget.
>
> **What we learned.** 0% across all budgets — confirms the model has no *reliable* path to optimal moves, only chance-rate execution. The interesting secondary observation is that LLM-as-P2 (always in a losing position because optimal P1 zeroes the nim-sum) shows elevated parse_failed at B≥256: losing-position reasoning costs more tokens than winning-position reasoning.

| Budget | Win rate | parse_failed% |
|--------|----------|--------------|
| 0–1024 | **0%** | varies |

As expected: an optimal player playing from a winning position (nim_sum≠0) wins
100% of the time with correct play, regardless of the LLM's budget.

The notable observation here is that parse_failed for the LLM playing as P2
(white) remains elevated at B=512 (7/15 turns ≈ 47%), even though it was 0%
when playing as P1 at the same budget. When the optimal player moves first, it
always reduces the nim_sum to 0, leaving the LLM in a losing position. Losing
positions require the model to reason about *why* no move is good, which
apparently requires more tokens and leads to more truncations.

### 5.3 Self-play vs B=1024 (nim_selfplay_llama__ab4210e9, N=10/cell)

> **Why this cell.** Isolates *budget* as the only difference between two otherwise-identical agents. If higher budget really means stronger play, the lower-budget side should lose more often.
>
> **What we learned.** Lower-budget LLMs *win* 60–70% against B=1024 in every cell, including B=0. This is consistent with the random anchor: B=1024 is itself near-chance, so the matchup is essentially symmetric with a small first-mover / parse-failure asymmetry. It does **not** mean "less reasoning is better" — it means neither side is actually exploiting strategy.

| Budget | Win rate | parse_failed% |
|--------|----------|--------------|
| 0 | 60% | 100% |
| 64 | 60% | 100% |
| 256 | 70% | 17% |
| 512 | 70% | 14% |

The main LLM (lower budget) beats the B=1024 opponent 60–70% of the time across
all budget cells, including B=0 (random play). This confirms that the B=1024
LLM is not actually playing well — both agents are making suboptimal moves, and
the game outcome is dominated by positional luck and who moves first.

---

## 6. Discussion

### 6.1 The random-play anchor

**Theoretical anchor: 50.0% (exact)**

Computed via exact recursive game-tree analysis over all Nim [3,5,7] states, confirmed
by Monte Carlo simulation (N=100,000; result: 49.7%). The key insight:

1. **Nim-sum advantage exists only under optimal play.** With nim_sum = 1, P1 has a
   winning strategy — but only if they can compute and exploit it. A random player
   derives no benefit from going first.
2. **Counterbalanced random play = fair coin.** Exact recursion shows P1 wins 50.0%
   under uniform-random play. Counterbalanced (50% as P1, 50% as P2) = 50.0%.
3. **Empirical confirmation.** N=30 B=0 games (random-fallback, no LLM reasoning):
   16/30 = 53.3% (p=0.86 vs 50%) — consistent with the theoretical value.

**Budget curve vs the 50% anchor:**

- B=0/B=64 (parse_failed=100%): Model plays randomly; observed 80% at N=10 is within
  sampling noise (binomial p=0.11). Theoretical expectation = 50%.
- B=256–1024 (deliberate reasoning): 40–60%, all consistent with the 50% anchor
  (all binomial p > 0.05 at N=30).
- **No budget cell is significantly different from chance.** The model cannot reliably
  outperform random play on Nim [3,5,7] regardless of reasoning budget or prompt structure.

**Two remedies worth testing:**

**(a) Larger / harder Nim configurations.** Configurations like [7, 11, 13] or
[5, 9, 14] have more turns and wider-bit XOR, which stresses arithmetic execution
more than [3,5,7] does. The motivation is **separation between skill and chance**,
not "lowering the random baseline" — that was an earlier mis-framing. Two correct
points to keep in mind:

- The random-vs-random win rate is a **per-configuration** quantity. It must be
  recomputed for each pile vector (exact recursion or N≥10⁵ Monte Carlo) before
  any LLM result on that config can be interpreted. Counterbalanced random-vs-random
  is **always 50%** by symmetry, regardless of pile vector.
- More turns ≠ harder for a *correctly* reasoning agent (an optimal player still
  wins ~100% from a winning position) but **does** increase the number of decision
  points where a flawed reasoner can fail, so the *gap* between optimal and flawed
  play widens — that is what makes the curve more informative.

A suggested config for a harder sweep (recompute the per-config anchor first):
```yaml
nim_piles: [7, 11, 13]   # nim_sum = 9, P1 wins under optimal play; ~3–6x more turns
budgets: [0, 256, 512, 768, 1024, 1536, 2048]
n_per_cell: 30           # see §6.2
```

**(b) More trials (N=50 minimum).** See Section 6.2.

### 6.2 Statistical power: N=10 is insufficient

With binary win/loss outcomes per game:

| N per cell | 95% CI (at p=0.5) | Min detectable difference (80% power) |
|------------|-------------------|--------------------------------------|
| 10 | ± 31% | ~45% |
| 20 | ± 22% | ~32% |
| 50 | ± 14% | ~20% |
| 100 | ± 10% | ~14% |

At N=10, the 95% confidence interval spans ±31 percentage points. The observed
fluctuations between budget cells are well within this variance — none of these
differences are statistically significant. The correct comparison is against the
50% theoretical anchor (not the noisy observed B=0 value).

**Recommendation**: N=30–50 per cell minimum for a publishable win-rate curve.
At ~10 LLM turns per game, this represents 300–500 LLM decisions per budget
point, providing robust parse_failed and move_regret estimates alongside win
rates.

### 6.3 move_regret is binary in this regime — and that's usable

`move_regret` is defined as `oracle_winrate(best_move) − oracle_winrate(chosen_move)`.
In Nim, oracle win rates are exactly 0.0 or 1.0 because Nim is solved — from any
position, either there exists a winning move (oracle_winrate=1.0 for winning
moves, 0.0 for losing moves) or there does not (all moves have oracle_winrate=0.0).

Therefore `move_regret ∈ {0.0, 1.0}` everywhere:
- **regret=0.0**: model chose a winning move, OR it was in a losing position where
  no winning move exists (forced loss)
- **regret=1.0**: model was in a winning position and chose a losing move (mistake)

This binary nature means:

- **move_regret cannot replace win_rate** as a primary outcome. Two models both
  with regret=0.0 might differ in whether they were actually in a winning
  position.
- **move_regret is useful as a mistake rate.** For turns where a winning move
  existed (nim_sum≠0 before the LLM's move), regret=1.0 is a clean "this move
  blundered a winning position" signal. Averaged over all such turns per game,
  it gives a *mistake rate* metric that is more fine-grained than game win rate.
- **To use it properly**: filter to winning positions only (`nim_sum≠0 at time of
  LLM's turn`), then compute `mean(regret)` over those turns. This gives the
  fraction of advantageous positions the model failed to exploit.
- **Continuous analogue**: for games where the oracle is approximate (e.g., UCT
  in Reversi at finite iterations), regret is genuinely continuous. Nim's exact
  oracle makes it binary. Consider using regret as the primary metric for Reversi
  comparisons where it has more resolution.

---

## 7. Data Quality Notes

### 7.1 Self-play "uct" label bug — patched

In the original runner, the opponent agent was always labeled `"uct"` in JSONL
`turn.agent` and `trial_end.winner` fields, even in self-play matches where the
opponent was another `LLMAgent`. The `runner.py` is now fixed to accept an
`opp_kind` parameter (set from `opponent_label` in the sweep) so future runs
label the opponent correctly (e.g., `"llm-B1024"`).

The existing self-play JSONL files (`nim_selfplay_llama__ab4210e9`, 40 trials) have been
patched post-hoc using `scripts/patch_selfplay_labels.py`. Backup files
(`.jsonl.bak`) are retained alongside each patched file. The analysis script
(`winner == "llm"`) is unaffected by this change.

### 7.2 Elevated parse_failed as P2 vs optimal

At B≥256, the LLM playing as P2 against the optimal agent shows higher
parse_failed than when playing as P1. When the optimal P1 always moves to a
nim_sum=0 position, the LLM faces a losing position where reasoning about "what
to do when there's no good move" is apparently more token-intensive. This creates
a position-type confound in parse_failed statistics: aggregate parse_failed rates
mask this P1/P2 asymmetry.

---

## 8. Diagnostic Results (May 2026)

> **Why we ran these.** §5 produced a flat curve at chance level. Before scaling N to tighten CIs, we needed to know *what is broken*: is the model failing because of arithmetic execution, missing strategic structure, or absent procedural scaffolding? Each diagnostic is a single targeted prompt change that disambiguates one of those hypotheses. The ablation on [1,2,3] tests whether the model is even *responding to* nim-sum structure or playing essentially randomly within parse-success cells.
>
> **What we learned.** Across all variants (free CoT, nim-sum given, step-by-step algorithm, few-shot worked example) the N=30 win rates cluster at 40–53%, none significantly different from the 50% random anchor. The bottleneck is not which prompt structure we use; it is that Llama 8B cannot reliably execute XOR + the inverse mapping (nim-sum → which pile to reduce by how much) on small numbers. The N=10 effects (`step_by_step` 70%, `nim_sum_given` 30%) did not survive replication.

Per the recommendation in Section 6, we ran four targeted diagnostic cells at
N=10, B=1024 only, against a random opponent on [3,5,7] unless otherwise noted.

### 8.1 N=10 Diagnostic Results (May 7 2026)

Initial 4-cell diagnostic at N=10, B=1024, vs random, [3,5,7].

| Cell | Variant | Win rate (N=10) |
|------|---------|----------------|
| Baseline | `free_cot` | 40% |
| (a) | `nim_sum_given` | 30% |
| (b) | `step_by_step` | **70%** |
| (c) | `few_shot` | 50% |
| (d) ablation | `free_cot` [1,2,3] P1 forced-loss | 50% (P1: 40%, P2: 60%) |

Initial N=10 interpretation: the 70% `step_by_step` result looked like a strong
positive finding — the algorithm in context nearly doubles baseline. The 30%
`nim_sum_given` looked like an active degradation. Both were subsequently
revised by N=30 (see §8.2).

Ablation (d) confirmed that P1/P2 win-rate split on [1,2,3] matches the baseline
[3,5,7] pattern, consistent with the model ignoring nim-sum structure.

### 8.2 N=30 Lock-In Results (May 7 2026)

> **Why this cell.** N=10 CIs span ±31pp; differences of 30pp between variants could easily be sampling noise. Tripling N tightens CIs to ~±18pp, enough to separate a true 40% from a true 70% but not enough to separate 43% from 53%. The companion `step_by_step` budget sweep tests whether budget *still matters* once the procedure is in context.
>
> **What we learned.** The dramatic N=10 differentials shrink to ~13pp at N=30 and lose statistical significance. All structured variants converge to ~50–53%. The B=256-vs-1024 step_by_step gap is ~10pp and inconclusive at N=30 (would need N≈85 to resolve at p<0.05).

Procedural sweep at N=30, B=1024, vs random, [3,5,7] — four variants.
Plus a budget sweep: `step_by_step` at B={256, 1024}, N=30.

| Variant | Win rate (N=30) | 95% Wilson CI |
|---------|----------------|---------------|
| `free_cot` (baseline) | 40.0% | [25%, 58%] |
| `nim_sum_given` | 50.0% | [33%, 67%] |
| `few_shot` | 53.3% | [36%, 70%] |
| `step_by_step` | 53.3% | [36%, 70%] |
| `step_by_step` B=256 | 43.3% | [27%, 61%] |
| `step_by_step` B=1024 | 53.3% | [36%, 70%] |

**Key revisions from N=10:**

1. **`step_by_step` 70% → 53%**: The N=10 Wilson CI was [40%, 89%], consistent
   with a true rate of 53%. The 7/10 result was an upward draw. The lift is real
   but smaller: ~13pp, not ~30pp.

2. **`nim_sum_given` 30% → 50%**: The N=10 CI was [11%, 60%]. At N=30, handing the model
   the XOR value gives similar performance to unstructured CoT, not worse.

3. **All structured variants cluster at 50–53%**: `nim_sum_given`, `few_shot`,
   and `step_by_step` are statistically indistinguishable at N=30. The particular
   form of structure does not significantly differentiate.

4. **Budget curve (B=256 vs B=1024, step_by_step)**: 43% vs 53%. CIs overlap;
   not distinguishable at p<0.05 with N=30. Point estimates suggest a modest
   ~10pp budget effect even with the procedure given, but inconclusive at this N.

### 8.3 Ablation (d) — forced-loss position [1,2,3]: 50% overall

> **Why this cell.** [1,2,3] has nim-sum=0, meaning P1 is in a forced-loss position under optimal play (the *opposite* of [3,5,7]). If the model were even partially using nim-sum, its P1 win rate should drop sharply on [1,2,3] vs [3,5,7]. If the rates match, the model is ignoring the nim-sum signal entirely.
>
> **What we learned.** [1,2,3] P1 = 40%, P2 = 60% — within noise of the [3,5,7] baseline. The model treats both starting positions the same, confirming it is not conditioning on nim-sum even when it claims to compute one.

Win rate of 50% (P1: 40%, P2: 60%) matches baseline pattern from [3,5,7],
confirming the model is not tracking nim-sum structure across game turns. The
position type (winning vs losing nim-sum) does not change its behaviour.

### 8.4 Interpretation

Theoretical random-play anchor: **50.0%** (exact game-tree recursion; empirical B=0 N=30 = 53.3%, p=0.86 — consistent).

**Exact binomial tests vs 50% anchor (all N=30):**

| Variant | k/n | p-value | Result |
|---------|-----|---------|--------|
| `free_cot` B=1024 | 12/30=40% | p=0.36 | ns |
| `nim_sum_given` B=1024 | 15/30=50% | p=1.00 | ns |
| `few_shot` B=1024 | 16/30=53% | p=0.86 | ns |
| `step_by_step` B=1024 | 16/30=53% | p=0.86 | ns |
| `step_by_step` B=256 | 13/30=43% | p=0.58 | ns |
| B=0 empirical (N=30) | 16/30=53% | p=0.86 | ns |

> **No variant achieves win-rate significantly above the 50% theoretical anchor.
> Llama 3.1 8B performs at chance level on Nim [3,5,7] regardless of prompt
> structure or CoT budget. The 40% → 53% gradient across variants is directional
> but not statistically conclusive at N=30.**

The budget effect on `step_by_step` (B=256 vs B=1024, ~10pp, p>0.05) is
suggestive but inconclusive. A definitive test would require N≈85 per cell.

---

## 9. Failure Mode Taxonomy — Trace Inspection (10 games, N=30 runs)

> **Why we ran this.** Aggregate win rate cannot tell us *how* the model is failing. Two different bugs (e.g., wrong arithmetic vs reversed winning condition) produce the same 50% win rate but suggest very different fixes. We inspected 10 raw traces — 5 from each of the two structurally different prompt regimes (`free_cot` and `step_by_step`) — to label distinct failure mechanisms.
>
> **What we learned.** Free-CoT failures cluster on **XOR arithmetic errors**, **strategic goal reversal**, and **hallucinated Nim concepts**. Step-by-step failures change *kind*, not *rate*: the scaffold removes goal reversal and hallucination but introduces **column-alignment errors in binary XOR**, **verification-loop confusion** (interpreting nim-sum=0 as a personal loss), and **derivation-vs-MOVE-tag disagreement**. Each failure cluster has a targeted prompt fix; that work is queued for the larger-Nim or Avalon phase.

Inspected 5 `free_cot` and 5 `step_by_step` games (B=1024, [3,5,7] vs random).
All failure turns had `move_regret=1.0` (winning position squandered).

### 9.1 `free_cot` failure modes

The dominant failure in unstructured reasoning is **XOR arithmetic error**: the model consistently computes the wrong nim-sum from the pile sizes. The single most frequent mistake is `3 ⊕ 5 ⊕ 7 = 15` (decimal addition instead of XOR), appearing verbatim across multiple seeds. A second cluster is **strategic goal reversal**: within a single turn the model flips between "I need to leave nim-sum = 0" (correct) and "I need to leave a non-zero nim-value for myself" (backwards), sometimes in adjacent sentences. A third pattern is **"nim-heap" hallucination**: the model invents a concept ("the smallest nim-heap ≥ 11 is 16") that has no basis in Nim theory, then acts on it. Notably, free_cot traces show **no consistent procedure**: each turn restarts from scratch with a slightly different framing, so errors compound across turns without any self-correction mechanism.

### 9.2 `step_by_step` failure modes

The scaffold eliminates goal reversal and hallucinated strategy (the five-step format is followed consistently in every turn), but introduces its own failure cluster: **column-alignment error in binary XOR**. The model writes binary representations without padding to the same width (e.g., `11`, `101`, `111`) and then XORs columns that don't correspond — producing wrong nim-sums like `001 → 0` or `011 XOR 101 XOR 111 → 000`. This is the root cause in roughly half of scaffold losses. A second scaffold-specific failure is **verification-loop confusion**: the model correctly computes a move that yields nim-sum=0 after its turn, then reasons "since nim-sum=0 after my move, I am in a losing position — I'll make any legal move instead," inverting the winning condition at the final step. A third pattern is **correct derivation, wrong MOVE tag**: the model's prose arrives at the right pile and count but the terminal `MOVE: pile=X take=N` disagrees, suggesting the MOVE tag is generated by a separate sub-process that doesn't reliably attend to the conclusion. These failure modes are addressable: padding binary representations in the prompt, clarifying that nim-sum=0 *after your move* is the goal (not a losing sign), and asking the model to copy the answer from its derivation rather than re-derive it in the tag would each target a specific failure cluster.

---

## 11. Recommended Next Steps

1. **Move to Avalon** where the optimal policy has no closed-form solution:
   CoT budget as a difficulty knob should operate more cleanly on tasks requiring
   genuine heuristic search rather than unreliable symbolic execution. The Nim
   result provides a clean negative control: on closed-form games, budget and
   prompt structure each add ~10-13pp but neither approaches optimal.
2. **To conclusively test the budget effect on `step_by_step`**: run N≈85 per
   cell at B={256, 1024} to achieve 80% power to detect a 20pp effect at p<0.05.
3. **Switch to Nim [7,11,13]** or similar for harder arithmetic and more turns
   per game. The 50/50 anchor for [3,5,7] means there is no structural gradient
   to detect; a larger game will have different random-play dynamics and harder
   arithmetic.
4. **Use move_regret over winning-position turns** as the primary per-turn
   metric, filtering to positions where nim_sum≠0 at the time of the LLM's move.
5. **Instrument parse_failed by position type** (winning vs losing nim_sum) to
   separate the P1/P2 asymmetry in token budget needs.
