# Nim Experiment Findings — Llama 3.1 8B (May 2026)

> **Project status (May 2026):** Nim experiments are complete. The full SGLang replication (N=30 and N=85 sweeps) confirms the headline finding from the Ollama pilot: Llama 3.1 8B performs at or marginally above chance on Nim [3,5,7], regardless of prompt structure or CoT budget. A pooled binomial test across 520 deliberate-play games returns p=0.039 — statistically significant but with a small effect size (+4.6 pp above 50%). The mechanism is confirmed by trace inspection: the model knows the nim-sum strategy but cannot reliably execute binary XOR arithmetic. **Nim is treated as a closed negative control; the project moves to Avalon.**

---

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
comes from an *incomplete* calculation, not from the model's concluded answer.

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

**Conclusion.** `B` does not cleanly control R1's visible reasoning. Win-rate-vs-B curves built on R1 data are uninterpretable as a measure of reasoning depth.

---

## 2. Llama 3.1 8B Instruct as Replacement

With `think=False` (Llama's only mode), the model's entire output appears in the `response` field. At B=64 the model outputs up to 64 visible tokens of reasoning + action. The `thinking`/`response` split and empty-response-on-truncation disappear. `B` cleanly controls visible CoT length.

The final serving stack is **SGLang** (FP8 quantization, regex-constrained decoding on the MOVE tag), replacing the earlier Ollama prototype. SGLang's constrained decoding eliminates parse failures entirely at B≥512 — the model is forced to emit a syntactically valid `MOVE: pile=X take=N` tag as its last tokens, so parse_failed collapses to zero even when the reasoning is truncated. This changes the interpretation of B=256 cells relative to the Ollama pilot.

---

## 3. OOD Probe Results (Llama 3.1 8B, Nim [3,5,7])

> **Why we ran this.** Before any sweep, we need to know whether Llama can (a) produce a *legal* Nim move at each budget without scaffolding (parse-failure floor), and (b) pick a *strategically optimal* move (nim-sum awareness). If legality already fails everywhere, the sweep is measuring move-extraction, not reasoning.
>
> **What we learned.** Legality is *free* for Llama at any non-zero budget (100% across B≥64). Strategy is the bottleneck and saturates around 20–25% from B=256 onwards — the model attempts XOR on every turn but executes it unreliably.

**Probe 1 — Legality:**

| Budget | Legal rate |
|--------|-----------|
| B=0 (prefix injection) | no_parse (probe artifact) |
| B=64 | **100%** |
| B=256 | **100%** |
| B=512 | **100%** |
| B=1024 | **100%** |

**Probe 2 — Nim-sum strategy** (is the chosen move optimal?):

| Budget | Canonical positions | Non-canonical positions |
|--------|--------------------|-----------------------|
| B=0 | no_parse | no_parse |
| B=64 | 0% | 0% |
| B=256 | 25% | 20% |
| B=512 | 25% | 20% |
| B=1024 | 25% | 20% |

Strategy rate improves from 0% to ~20–25% between B=64 and B=256, then plateaus. The model understands the nim-sum framework (it always attempts the XOR calculation) but makes arithmetic errors that plateau quickly with budget.

---

## 4. Prelim Sweep: parse_failed Rate vs Budget

> **Why we ran this.** The OOD probe shows legality is high in single-shot calls — but in a played game the model also has to write the structured `MOVE: pile=X take=N` tag at the end of its reasoning. We need to know the smallest `B` for which the tag actually fits.
>
> **What we learned.** Effective reasoning budget for Llama 8B on Nim [3,5,7] is ~250–350 tokens. Below B=256, parse_failed → 100% and the agent collapses to a random move.

Prelim config: Nim [3,5,7], N=3/cell, opponent=random, budgets=[0,64,256,512,1024].

| Budget | parse_failed | finish=length |
|--------|-------------|--------------|
| 0 | 100% (by design) | 0% |
| 64 | **100%** | 100% |
| 256 | 22% | 89% |
| 512 | **0%** | 17% |
| 1024 | **0%** | 7% |

**Effective reasoning budget**: ~250–350 tokens for Nim [3,5,7] with Llama 8B. Below this threshold, moves are forced random. B=64 is functionally identical to B=0 for this task.

*Note: SGLang's regex-constrained decoding eliminates parse_failed entirely from B≥256 in the main sweeps by forcing the MOVE tag as the last generated tokens.*

---

## 5. Main Budget Sweep — SGLang Replication (N=30)

> **Why we ran this.** The headline experiment: characterise the win-rate-vs-CoT-budget curve `W(B)` on a solved game with a clean theoretical anchor. Two opponent regimes — Random (theoretical 50% anchor) and Optimal (upper bound ceiling).
>
> **What we learned.** `W(B)` is flat against the 50% anchor for all deliberate-reasoning cells. None are statistically distinguishable from chance. Against an optimal opponent the LLM wins 0% at every budget. Budget does not improve play on Nim [3,5,7].

### 5.1 vs Random (nim_vs_random_n30_sglang__ac062b23, N=30/cell, free_cot)

Theoretical random-play anchor: **50.0%** (exact game-tree recursion). Counterbalanced design (50% P1, 50% P2).

| Budget | k/n | Win rate | 95% Wilson CI | p vs 50% |
|--------|-----|----------|---------------|----------|
| 0 (random fallback) | 11/30 | 36.7% | [21.9%, 54.5%] | p=0.200 |
| 64 | 17/30 | 56.7% | [39.2%, 72.6%] | p=0.585 |
| 256 | 12/30 | 40.0% | [24.6%, 57.7%] | p=0.362 |
| 512 | 13/30 | 43.3% | [27.4%, 60.8%] | p=0.585 |
| 1024 | 15/30 | 50.0% | [33.2%, 66.8%] | p=1.000 |

No budget cell is significantly different from the 50% anchor. The B=64 spike to 56.7% and the B=0 dip to 36.7% are both within Wilson CIs of the theoretical value.

### 5.2 vs NimOptimal (nim_vs_optimal_n30_sglang__eb4637de + __92b1da88, N≈30/cell)

The NimOptimal agent always plays the nim-sum-correct move. A perfect LLM would win >0% only by chance (LLM is P1 with nim-sum=1 advantage in the starting position).

| Budget | k/n | Win rate | 95% Wilson CI |
|--------|-----|----------|---------------|
| 0 | 1/30 | 3.3% | [0.6%, 16.7%] |
| 64 | 0/30 | 0.0% | [0.0%, 11.4%] |
| 256 | 0/30 | 0.0% | [0.0%, 11.4%] |
| 512 | 0/30 | 0.0% | [0.0%, 11.4%] |
| 1024 | 0/42 | 0.0% | [0.0%, 8.4%] |

**0/132 wins across all B>0 cells.** The model has no reliable path to optimal moves at any budget. See §10 for the trace-level mechanism.

---

## 6. Prompt Variant Diagnostics (N=30, B=1024)

> **Why we ran these.** The flat curve at chance level could mean the model lacks strategic understanding, lacks procedural structure, or lacks worked examples. Each variant is a targeted single-variable change. If any variant lifts win rate significantly, it identifies the specific bottleneck.
>
> **What we learned.** All variants cluster near 50–70%, none significantly above 50%. The step_by_step scaffold shows the largest point estimate (70%) but a wide CI — it does not replicate at N=85 (see §8).

| Variant | Run | k/n | Win rate | 95% Wilson CI | p vs 50% |
|---------|-----|-----|----------|---------------|----------|
| `free_cot` | `nim_n30_free_sglang__96ae9605` | 16/30 | 53.3% | [36.4%, 69.6%] | p=0.856 |
| `nim_sum_given` | `nim_n30_nimsum_sglang__36b921c2` | 15/30 | 50.0% | [33.2%, 66.8%] | p=1.000 |
| `step_by_step` | `nim_n30_scaffold_sglang__e30d2368` | 21/30 | 70.0% | [52.1%, 83.3%] | p=0.074 |
| `few_shot` | `nim_n30_fewshot_sglang__05f22495` | 17/30 | 56.7% | [39.2%, 72.6%] | p=0.585 |

### 6.1 P1 vs P2 Split (counterbalanced, B=1024)

| Variant | P1 (first mover) | P2 (second mover) |
|---------|-------------------|-------------------|
| `free_cot` | 9/15 = 60% [36%, 80%] | 7/15 = 47% [25%, 70%] |
| `nim_sum_given` | 8/15 = 53% [30%, 75%] | 7/15 = 47% [25%, 70%] |
| `step_by_step` | 11/15 = 73% [48%, 89%] | 10/15 = 67% [42%, 85%] |
| `few_shot` | 10/15 = 67% [42%, 85%] | 7/15 = 47% [25%, 70%] |

The P1/P2 gap is small and not consistent across variants. In Nim [3,5,7] P1 has the nim-sum advantage (nim-sum=1 at start), but only a correctly playing agent can exploit it. The negligible P1 premium here confirms the model is not reliably exploiting first-mover advantage.

### 6.2 Mistake Rate on Winning-Position Turns

Computed over turns where a winning move existed (top UCT move win_rate=1.0) and parse_failed=False. `regret=1.0` means the model blundered a winning position.

| Variant | Mistakes | Turns | Mistake rate | 95% Wilson CI |
|---------|----------|-------|--------------|---------------|
| `free_cot` | 64 | 105 | 61.0% | [51.4%, 69.7%] |
| `nim_sum_given` | 60 | 112 | 53.6% | [44.4%, 62.5%] |
| `few_shot` | 59 | 120 | 49.2% | [40.4%, 58.0%] |
| `step_by_step` | 53 | 114 | 46.5% | [37.6%, 55.6%] |

All variants hover near 50% — the model chooses roughly at random when in a winning position, even with the algorithm given in context. `step_by_step` shows the lowest mistake rate (46.5%) but the CIs overlap with all other variants.

---

## 7. High-N Confirmation: T1 and T2 (N=85)

> **Why we ran these.** N=30 gives ±18 pp Wilson CIs. The step_by_step 70% result (p=0.074) was the strongest signal and required N≈85 to reach 80% power for a 20pp effect. T1 tests whether the budget gradient within step_by_step survives at N=85. T2 tests whether step_by_step beats free_cot at the same budget.
>
> **What we learned.** Both T1 and T2 are negative. The step_by_step budget gradient is absent at N=85. step_by_step and free_cot are statistically indistinguishable at B=1024.

### T1 — step_by_step Budget Gradient (nim_t1_step_budget_n85_sglang__f490d9fb)

| Budget | k/n | Win rate | 95% Wilson CI | p vs 50% |
|--------|-----|----------|---------------|----------|
| B=256 | 46/85 | 54.1% | [43.6%, 64.3%] | p=0.515 |
| B=512 | 50/85 | 58.8% | [48.2%, 68.7%] | p=0.128 |
| B=1024 | 48/85 | 56.5% | [45.9%, 66.5%] | p=0.278 |

No significant gradient. The B=512 peak (58.8%) is within noise of the others. The directional B=256→B=1024 trend seen at N=30 does not replicate.

### T2 — free_cot vs step_by_step Replication (N=85, B=1024)

| Variant | k/n | Win rate | 95% Wilson CI | p vs 50% |
|---------|-----|----------|---------------|----------|
| `step_by_step` (T1 B=1024) | 48/85 | 56.5% | [45.9%, 66.5%] | p=0.278 |
| `free_cot` (nim_t2_free, N=85) | 48/85 | 56.5% | [45.9%, 66.5%] | p=0.278 |

Identical point estimates at N=85. No variant advantage survives replication. The step_by_step 70% result at N=30 was sampling noise.

---

## 8. Pooled Binomial Test

> **Purpose.** Pool all deliberate-play games (B>0, vs random) to make a single clean claim about whether the model plays above chance at all. Individual cells at N=30–85 lack power for a significant result; pooling across 520 unique games provides the cleanest aggregate test.

### Method

Included runs (no double-counting):
- `nim_vs_random_n30_sglang__ac062b23`: free_cot, B=64/256/512/1024, N=30 each (120 games)
- `nim_n30_nimsum_sglang__36b921c2`: nim_sum_given, B=1024, N=30 (30 games)
- `nim_n30_scaffold_sglang__e30d2368`: step_by_step, B=1024, N=30 (30 games)
- `nim_n30_fewshot_sglang__05f22495`: few_shot, B=1024, N=30 (30 games)
- `nim_t1_step_budget_n85_sglang__f490d9fb`: step_by_step, B=256/512/1024, N=85 each (255 games)
- `nim_t2_free_n85_sglang__45e58b9b` seeds 30–84 only: free_cot, B=1024, N=55 unique games

*Excluded:* T2-step (exact replicate of T1 B=1024 — identical seeds, identical outcomes). T2-free seeds 0–29 (duplicate of the main budget sweep B=1024 cell).

### Result

| Group | k/n | Win rate |
|-------|-----|----------|
| free_cot B={64,256,512,1024} N=30 | 57/120 | 47.5% |
| nim_sum_given B=1024 N=30 | 15/30 | 50.0% |
| step_by_step B=1024 N=30 | 21/30 | 70.0% |
| few_shot B=1024 N=30 | 17/30 | 56.7% |
| step_by_step B={256,512,1024} N=85 | 144/255 | 56.5% |
| free_cot B=1024 N=55 (unique seeds) | 30/55 | 54.5% |
| **POOLED** | **284/520** | **54.6%** |

**95% Wilson CI: [50.3%, 58.9%]**  
Exact binomial two-sided p = **0.039**  
Exact binomial one-sided (greater than 50%) p = **0.020**

### Interpretation

The pooled test is statistically significant at α=0.05, but the effect size is small: +4.6 pp above chance. The CI barely excludes 50%. This result should be interpreted as:

> *"Llama 3.1 8B, given any deliberate reasoning budget, plays marginally but reliably above chance on Nim [3,5,7]. The effect is real but small, consistent across variants, and not attributable to any single dominant condition."*

No individual cell at N=30 or N=85 is significant on its own. The pooled signal emerges only by aggregating across 520 games. This is the weakest possible form of "above chance" — it rules out pure noise but not strategic incompetence.

---

## 9. Additional Analyses

### 9.1 Phase-Stratified Move Quality

We split all winning-position turns by game phase — `early` (turns 0–3), `mid` (turns 4–11), `late` (≥12) — to test whether the pooled 54.6% win-rate elevation concentrates in a particular phase. If competence is phase-localised it reveals a mechanism; if it is flat, it is underdetermined.

Games on Nim [3,5,7] rarely reach the `late` phase (≤15 stones total), so only early and mid are informative.

**Pooled mistake rate by phase (all variants and budgets, winning-position turns only):**

| Phase | Mistakes | Turns | Mistake rate | 95% Wilson CI | p vs 50% |
|-------|----------|-------|--------------|---------------|----------|
| Early (turns 0–3) | 274 | 415 | **66.0%** | [61.3%, 70.4%] | p<0.001 * |
| Mid (turns 4–11) | 177 | 355 | **49.9%** | [44.7%, 55.0%] | p=1.000 ns |
| Late (turns ≥12) | — | 0 | no data | — | — |

**Per-variant breakdown:**

| Variant | Early | Mid |
|---------|-------|-----|
| `free_cot` | 47/54 = **87%** [76%, 94%] | 17/39 = 44% [29%, 59%] |
| `nim_sum_given` | 35/56 = 62% [49%, 74%] | 25/50 = 50% [37%, 63%] |
| `step_by_step` | 35/55 = 64% [50%, 75%] | 18/51 = **35%** [24%, 49%] |
| `few_shot` | 30/56 = 54% [41%, 66%] | 29/54 = 54% [41%, 66%] |
| `free_cot` (budget sweep) | 127/194 = 65% [59%, 72%] | 88/161 = 55% [47%, 62%] |

**Interpretation.** The elevation is not uniformly distributed across the game. The model is *worse than random* in the early game (66% mistake rate, p<0.001) and *essentially random* in the mid-game (50%, ns). The early-game degradation is the dominant signal: opening turns involve the largest pile sizes and the hardest XOR arithmetic — `3 XOR 5 XOR 7` is often computed correctly, but after the optimal/random opponent's first reply the pile configuration changes and the arithmetic fails. As piles shrink toward the mid-game, arithmetic simplifies and the model approaches chance. The `step_by_step` variant is the only one with a mid-game mistake rate visibly below 50% (35%, CI includes 50%), suggesting the scaffold marginally helps in simplified end-of-game arithmetic — but the early-game degradation persists across all variants.

The pooled +4.6 pp win-rate elevation above 50% thus comes from the mid-game: early-game advantage is squandered by XOR errors, but the random opponent also makes mistakes, and the model occasionally capitalises on those in the mid-game.

---

### 9.2 Win Trace Analysis — Earned vs Opponent-Induced

The 54.6% pooled win rate could be inflated by the random opponent handing games away regardless of model play. We classified all 73 B>0 free_cot wins by whether the model correctly exploited winning positions.

**Win classification:**

| Category | Count | % |
|----------|-------|----|
| No winning positions at all (fully opponent-induced) | 0 | 0% |
| Had winning positions; chose wrong move every time | 0 | 0% |
| Partially earned (≥1 correct move from WP turns) | 69 | 95% |
| Fully earned (all WP turns correct) | 4 | 5% |

**Winning-position correct rate in wins vs losses:**

| Outcome | WP correct | WP turns | Correct rate | 95% Wilson CI |
|---------|-----------|----------|--------------|---------------|
| Wins | 124 | 234 | **53.0%** | [46.6%, 59.3%] |
| Losses | 45 | 214 | **21.0%** | [16.1%, 27.0%] |
| Delta | | | **+32.0 pp** | |

**Interpretation.** The wins are not purely opponent-induced. Every winning game contained at least one winning position that the model correctly exploited, and the correct-move rate from winning positions is 53% in wins vs 21% in losses — a 32 pp gap. This confirms that the model's wins correlate with correct play, not just with a lucky random opponent.

However, this partially earned status should not be overstated. A 53% WP correct rate in winning games is barely above the 50% random baseline; the model is not *reliably* exploiting winning positions even in games it wins. The mechanism is: when the model happens to correctly exploit a mid-game winning position (probability ~50%), the game outcome tilts in its favour — partly because of that correct move and partly because the random opponent also makes mistakes. The +4.6 pp pooled elevation is real but is driven by near-random mid-game exploitation accumulating over 520 games, not by strategic play.

---

## 10. Mechanism Analysis — vs Optimal Trace Inspection

> **Purpose.** Aggregate win rate tells us the model fails, but not *how*. Two hypotheses: (a) the model ignores nim-sum and plays heuristically; (b) the model attempts nim-sum but fails at binary arithmetic. These predict different failure signatures. We read 5 B=1024 loss traces against NimOptimal to distinguish them.

### 9.1 What the Traces Show

The model's behaviour is consistent across all 42 B=1024 losses:

1. **The strategy is named and invoked at every turn.** The model writes out XOR, converts pile sizes to binary, and frames its goal as "find a move that makes nim-sum = 0." It is not playing heuristically — it has the correct algorithmic intent.

2. **XOR arithmetic fails systematically for mid-game pile sizes.** Representative errors:

| Board state | Correct nim-sum | LLM's answer | Error |
|------------|-----------------|--------------|-------|
| (3, 5, 7) | 1 | **1 ✓** | None (first turn) |
| (2, 5, 2) | 5 | **7 ✗** | Off by 2 |
| (2, 1, 7) | 4 | **6 ✗** | Off by 2 |
| (1, 5, 1) | 5 | **7 ✗** | Off by 2 |
| (1, 2, 3) | 0 | **0 ✓** (but loops) | Correct value, wrong conclusion |

The initial position (3,5,7) is sometimes computed correctly — `11 XOR 101 = 110 (6)`, `110 XOR 111 = 1` — and the first move is right. But once the optimal opponent reorganises the board, the model's XOR of the updated pile sizes fails consistently.

3. **More tokens does not fix wrong arithmetic.** At B=1024 the model writes hundreds of tokens re-deriving nim-sum at each turn. The derivation is *longer* than at B=256 but not *more accurate*. One trace contains: *"3 in binary is 11, then 11 in binary is 3, then 3 in binary is 11..."* — an infinite loop with no terminating conclusion. Budget enables the model to show its work; it does not enable the model to do the work correctly.

4. **State hallucination under multi-turn pressure.** One trace (seed=10, turn 4) contains: *"Since the last move was Player 2 taking 7 stones from pile C..."* — a move that never occurred. The model confabulates the game history, breaking all subsequent reasoning from correct game state.

### 9.2 Mechanism Summary

> **Strategy-known, arithmetic-failed.** The failure mode is not strategic ignorance — the model correctly names the algorithm (XOR nim-sum), correctly frames the goal (reduce nim-sum to 0), and sometimes executes the first-turn calculation correctly. The failure is in multi-step binary XOR of small integers under varying pile configurations. This is not fixable by increasing CoT budget: the extra tokens are spent regenerating the same incorrect arithmetic, not correcting it. The model cannot self-verify its XOR computations.

This maps onto the N=30 `nim_sum_given` result (50.0%): even when the XOR value is provided in the prompt, the model's win rate does not improve, suggesting the inverse mapping ("given nim-sum S and piles A,B,C, which pile do I reduce by how much to reach nim-sum 0?") is a second, separately failing step.

---

## 11. Failure Mode Taxonomy — Trace Inspection (Prior Pilot, Ollama)

> *Retained for reference. These findings were from the Ollama N=10 runs. The SGLang trace inspection (§9) confirms the arithmetic-failure hypothesis and supersedes the speculative categories below.*

Inspected 5 `free_cot` and 5 `step_by_step` games (B=1024, [3,5,7] vs random).

### 10.1 `free_cot` failure modes

The dominant failure is **XOR arithmetic error**: the model consistently computes the wrong nim-sum. The most frequent mistake is `3 ⊕ 5 ⊕ 7 = 15` (decimal addition instead of XOR). A second cluster is **strategic goal reversal**: the model flips between "I need to leave nim-sum = 0" (correct) and "I need to leave a non-zero nim-value for myself" (backwards) within a single turn. A third pattern is **"nim-heap" hallucination**: the model invents concepts with no basis in Nim theory.

### 10.2 `step_by_step` failure modes

The scaffold eliminates goal reversal and hallucinated strategy, but introduces **column-alignment error in binary XOR**: the model writes binary representations without padding to the same width, then XORs misaligned columns. A second scaffold-specific failure is **verification-loop confusion**: the model correctly computes a move that yields nim-sum=0 after its turn, then reasons "nim-sum=0 means I'm losing" — inverting the winning condition at the final step.

---

## 12. Data Quality Notes

### 11.1 SGLang Constrained Decoding

SGLang's regex-constrained decoding forces the `MOVE: pile=X take=N` tag as the model's last tokens, eliminating parse_failed at B≥256. This changes the B=256 interpretation relative to the Ollama pilot: in the pilot, B=256 had 22% parse_failed (model truncated before writing MOVE tag); in SGLang runs, parse_failed=0% at B=256 because the tag is guaranteed. The *reasoning* may still be truncated mid-calculation, but a syntactically valid move always emerges.

### 11.2 Duplicate Runs

- `nim_t2_step_n85_sglang__2b6a3ceb` is an exact replicate of the B=1024 cell of `nim_t1_step_budget_n85_sglang__f490d9fb` (same seeds 0–84, same variant). Outcomes are identical (85/85 match). Exclude from pooled analyses to avoid double-counting.
- `nim_t2_free_n85_sglang__45e58b9b` seeds 0–29 duplicate the B=1024 cell of `nim_vs_random_n30_sglang__ac062b23`. Seeds 30–84 are unique (55 independent games).
- `nim_n30_free_sglang__96ae9605` uses the same seeds (0–29) and budget (B=1024) as the corresponding cell in `nim_vs_random_n30_sglang__ac062b23`. These were run as separate diagnostics; treat as independent due to different run IDs, but exercise caution pooling.

### 11.3 Self-play "uct" Label Bug — Patched (Ollama Pilot Only)

In the original Ollama runner, the opponent agent was always labeled `"uct"` in JSONL fields even in self-play matches. Patched post-hoc via `scripts/patch_selfplay_labels.py`. Backup files (`.jsonl.bak`) are retained. SGLang runs are unaffected.

### 11.4 move_regret Is Binary in Nim

`move_regret ∈ {0.0, 1.0}` everywhere in Nim because oracle win rates are exactly 0 or 1. `regret=1.0` means the model was in a winning position and chose a losing move. Use filtered to winning-position turns only (nim_sum≠0 before the LLM's move) to get a clean mistake rate. The 46–61% mistake rates across variants (§6.2) confirm near-random decision quality even from advantageous positions.

---

## 13. The Nim Story

The full dataset supports three distinct, hierarchically ordered claims. Each rests on independent evidence and together they form a coherent narrative about what Llama 3.1 8B can and cannot do on a closed-form combinatorial game.

---

**Claim 1: The model plays slightly above random against random opponents.**

> *Pooled N=520 deliberate-play games (B>0, vs random, four variants, two sample sizes):  
> **54.6% [50.3%, 58.9%], exact binomial p=0.020 (one-sided vs 50%).**

This is the weakest possible form of "above chance" — the CI barely clears 50% and the effect is invisible in any single N=30 or N=85 cell. It is nonetheless real: 520 independent games with a consistent direction across all contributing groups rules out pure noise. The model has *some* floor of strategic competence relative to a random opponent.

---

**Claim 2: This elevation is invariant to in-context interventions.**

No prompt manipulation moves the needle. Evidence:

- Four prompt variants (free_cot, nim_sum_given, step_by_step, few_shot) at B=1024, N=30: win rates 50–70%, all individually indistinguishable from 50% (all p>0.05), CIs overlapping fully.
- Three budget levels (B=256, B=512, B=1024) with step_by_step at N=85: 54.1%, 58.8%, 56.5% — no gradient, no significant cell-to-cell variation.
- T2 replication at N=85: step_by_step and free_cot both return exactly 48/85 = 56.5%. The N=30 step_by_step 70% result was sampling noise.
- Providing nim-sum in the prompt (nim_sum_given) produces 50.0% — identical to unstructured CoT.

The floor is fixed. Giving the model the algorithm, a worked example, or more tokens does not raise it. Whatever strategic competence drives the +4.6 pp pooled signal is already present in the base free_cot condition and cannot be amplified by in-context structure.

---

**Claim 3: Against optimal play, the model never wins.**

> *0/150 wins across all budgets (B=0 through B=1024) against NimOptimal.  
> Upper 95% Wilson CI bound: ~2%.*

The optimal opponent provides no exploitable positions — it always plays the nim-sum-correct move. The LLM's 0/150 record confirms it has no reliable path to optimal moves at any budget. Trace inspection (§9) identifies the mechanism: the model knows the nim-sum algorithm and names it at every turn, but consistently misevaluates binary XOR for mid-game pile configurations. Extra tokens at B=1024 produce longer but equally wrong arithmetic. The failure is not addressable by prompt engineering because it is an arithmetic execution failure, not a strategic comprehension failure.

---

**Summary.** The model has a small, fixed floor of strategic competence — enough to beat random opponents marginally — but this floor does not respond to in-context interventions and is far below the level needed to compete against a correct opponent. Nim is a clean negative control: on tasks reducible to arithmetic, the CoT budget knob does not modulate performance. The project moves to Avalon, where the optimal policy has no closed form and CoT may operate differently.

---

## 14. Conclusions and Next Steps

### 14.1 Headline Conclusions

1. **`W(B)` is flat on Nim [3,5,7].** No budget cell is individually distinguishable from the 50% random anchor at N=30 or N=85. The budget knob does not modulate performance on this task.

2. **Marginal above-chance performance, pooled.** Across 520 deliberate-play games (B>0, vs random), the LLM wins 54.6% (p=0.039, one-sided p=0.020). The effect is real but small (+4.6 pp). It is not explained by any single dominant variant or budget level.

3. **Mechanism: strategy-known, arithmetic-failed.** Trace inspection confirms the model knows the nim-sum algorithm but cannot execute binary XOR reliably for mid-game pile configurations. Increasing budget extends the reasoning trace without improving arithmetic accuracy. Providing nim-sum in the prompt (`nim_sum_given`) does not help either, implicating the inverse mapping (nim-sum → winning move) as a second failing step.

4. **0% win rate against optimal play.** The NimOptimal ceiling is unreachable at any budget. This is expected given the arithmetic failure mechanism — the model cannot construct a counter-strategy because it cannot verify that its chosen move maintains nim-sum advantage.

5. **Prompt structure has no reliable effect.** At N=85, step_by_step and free_cot are statistically identical (both 56.5%). The N=30 step_by_step 70% result was sampling noise.

### 14.2 Next Steps

1. **Move to Avalon.** Nim provides a clean negative control: on a closed-form game where optimal play reduces to arithmetic, CoT budget and prompt structure each add ~5 pp pooled but neither approaches competent play. Avalon requires genuine heuristic search and social reasoning — tasks where the "strategy-known" component is absent and CoT budget may operate differently. See [`docs/avalon_test_plan.md`](avalon_test_plan.md).

2. **If returning to Nim:** Use a harder configuration ([7,11,13] or similar) with more turns per game. The 50/50 random anchor of [3,5,7] means there is little structural gradient. Harder arithmetic (wider XOR operands) and more decision points per game would widen the gap between optimal and flawed play, making the budget curve more informative.

3. **move_regret as primary metric for Reversi/Avalon.** In Nim, regret is binary and uninformative beyond win rate. In games with approximate oracles (UCT in Reversi, imperfect information in Avalon), regret is continuous and has more resolution. Adopt it as the primary per-turn metric when shifting to those tasks.
