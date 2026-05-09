# CoT Budget as a Compute-Grounded Difficulty Knob for LLM Game Agents

**Stephone [Last Name]** — [sunetid]@stanford.edu

CS 348K — Visual Computing Systems, Spring 2026

---

> **Project status (May 2026).** The active workstream is a **Nim pilot on Llama 3.1 8B Instruct** (Ollama, single-pass `think=False`), not the Reversi/R1/Ludii stack described below. Reasons: (1) DeepSeek R1's `thinking`/`response` split is incompatible with treating `B` as a *visible* CoT budget, and (2) Reversi OOD probes show R1-Distill-7B cannot identify legal moves without a Pass-2 legal-move list (see [`docs/ood_probe_findings.md`](ood_probe_findings.md)), so legality — not strategy — would dominate the curve. The Nim pilot's results, anchors, and diagnostics are in [`docs/nim_experiment_findings.md`](nim_experiment_findings.md). The Reversi/Ludii target stack remains the planned end state; see [`docs/architecture.md`](architecture.md) for the current-vs-target mapping.

---

## Summary

We will produce a set of win-rate-vs-CoT-budget curves characterizing how chain-of-thought (CoT) token budget controls the difficulty of an LLM-based game-playing agent on Reversi, accompanied by a reasoning-content analysis that validates the additional tokens are being used for substantive reasoning operations rather than verbosity, plus a closed-loop demonstration of an adaptive difficulty controller that adjusts CoT budget online to track a target win rate against opponents of varying skill. The approach uses DeepSeek-R1-Distill-Qwen-7B INT4 served via SGLang with constrained decoding, runs self-play against UCT baselines through the Ludii Java API, and isolates reasoning content from token-count artifacts via an arithmetic filler control. As a preliminary generalization test, we will run a small budget sweep on Avalon (using AvalonBench infrastructure) to verify the methodology produces a coherent signal in a game where UCT does not apply.

---

## Inputs and Outputs

**System inputs.** A game G (Reversi 8×8 as the primary case; Avalon as the preliminary generalization case), a CoT token budget B ∈ {0, 64, 256, 1024} for static curves and B ∈ {0, 32, 64, 128, 256, 512, 1024, 2048} for the controller, a model configuration (DeepSeek-R1-Distill-Qwen-7B INT4 primary; Llama 3.2 3B Instruct as a non-reasoning comparison; Qwen 2.5 1.5B Instruct as a small-model comparison), a memory mode (structured summary vs. full chronological history), and an opponent specification (UCT-2000 primary, UCT-500 and UCT-10000 for controller robustness).

**System outputs.** Win rate W(B, G, model) over N=50 self-play trials per condition with 95% bootstrap confidence intervals, move entropy H(B, G), per-turn move quality logs (agreement with UCT top-3 picks), reasoning transcripts for all decisions, and adaptive controller time series (B over game index, observed win rate over game index, target line).

**Constraints.** The fundamental constraint is inference throughput. Game decisions must be fast enough to run thousands of trials within a six-week timeline on local hardware (RTX 5080 + Quadro RTX 6000), which drives the choice of 4-bit quantized 7B models and a max budget of 1024 tokens for primary experiments. A secondary constraint is confound isolation: reasoning quality must be distinguished from output-length artifacts via an explicit matched-length filler control. A third constraint is reproducibility: all experiments run locally with no API calls.

**The crux.** Three layers of distinction matter for this work, and conflating them leads to invalid claims.
First, *output token budget* (B) is what we control directly via `max_tokens=B`.
Second, *win rate* is a noisy outcome metric that we measure, calibrated against UCT for ground-truth strength.
Third, *reasoning content quality* is what the budget enables but does not guarantee — more tokens could be spent on substantive reasoning operations or on verbose restatement.

We do not claim that more tokens always produces better reasoning. We claim that under controlled conditions, allocating more tokens *enables* the model to produce better outcomes on average, with the reasoning content analysis providing direct evidence of how the additional tokens are being used. Game difficulty itself is not directly observable; it must be inferred from win-rate distributions across many stochastic trials. Isolating reasoning content as the mechanism (vs. token-count artifacts like sampling diversity, recency bias, or move commitment) requires a matched-length filler baseline using game-content-free arithmetic tokens, plus a memorization probe (sampling at T=0.7 on UCT-flagged uncertain positions) to verify low-budget wins are not pattern-matched from training data.

---

## Task List

**CoT prompt structure.** The agent receives a structured-JSON game state, an enumerated list of legal moves, a structured summary of recent history, and a free-text reasoning prompt. The default reasoning prompt is unstructured ("Think through your move.") to avoid confounding budget effects with prompt structure effects. We compare this against an explicitly structured prompt in Task 8.

```
[System prompt: rules of Reversi, format requirements]

Current game state:
{json state with board, scores, turn number}

Legal moves:
{enumerated list, e.g., "0: c4, 1: d3, ..."}

Recent history:
{structured summary, ≤256 tokens}

Think through your move.

Respond with your reasoning in <reasoning>...</reasoning> tags.
After your reasoning, you will be asked to select a move.
```

Pass 2 (move selection) is conditioned on Pass 1's output and constrained via SGLang's `choices` operator to produce exactly one legal move token. We deliberately do *not* borrow ReAct, Tree-of-Thought, or other established CoT scaffolds — those are different methods, not different budgets, and their inclusion would confound budget effects with method effects.

**Tasks I will implement.** The following list is what I expect to complete to earn a desirable grade.

1. *Stack validation and end-to-end pipeline (week 1).* Stand up SGLang serving DeepSeek-R1-Distill-Qwen-7B INT4 on the RTX 5080. Test JPype Java↔Python bridge under concurrent async load (known issue area). Implement Reversi `GameSerializer` and `MoveSerializer`. Get one full Reversi self-play game running end-to-end against Ludii's UCT-2000 baseline with the two-pass generation design (Pass 1: free reasoning with `max_tokens=B`; Pass 2: constrained move selection via SGLang's `choices` operator). End of week 1 target: one full game runs reliably.

2. *Cross-game batching for throughput (week 2).* Wrap single-game pipeline in asyncio coroutines. Tune SGLang's `max_running_requests` and `mem_fraction_static` to find the right concurrency level (target 8–16 games per GPU). Verify SGLang's RadixAttention prefix caching is firing by placing static prompt content first. Profile actual throughput against the ~80 tok/sec single-stream baseline; expect ~400–500 tok/sec aggregate at 8-way batching.

3. *Primary budget sweep on Reversi (weeks 2–3).* Run N=50 self-play games against UCT-2000 for each (B, model) cell. Budget levels B ∈ {0, 64, 256, 1024}. Models: R1-Distill-7B (primary) and Llama 3.2 3B (non-reasoning comparison). Memory: structured summary, greedy decoding (T=0). Output: W(B, model) curves with 95% bootstrap CIs. This is the main result.

4. *Confound isolation experiments (week 3).*
   - *Arithmetic filler control:* at each B, replace the reasoning trace with B tokens of pre-generated arithmetic problem-and-answer text (game-independent, length-matched at the token level using the agent's tokenizer). N=30 per cell. If filler win rate matches B=0 baseline rather than full CoT, reasoning is established as the mechanism.
   - *Memorization probe:* identify ~50 game states where UCT sees multiple roughly-equal moves (uncertain positions). At each, sample 10 moves at T=0.7, B=0. High consistency on uncertain positions is a memorization signature.
   - *Constrained decoding ablation:* N=20 at B=256, comparing SGLang's `choices` constraint vs. unconstrained generation with post-hoc parsing.

5. *Memory design comparison (week 4).* Two conditions on Reversi at B=256: structured summary vs. full chronological history. N=30 per condition. Sub-analysis: summary stability check (BLEU and embedding similarity of summaries produced for the same game state across budget conditions) to verify the summarizer behaves consistently and is not a budget-correlated confound.

6. *Model size comparison (week 4).* Add Qwen 2.5 1.5B Instruct on Reversi. Full B sweep, N=50 per condition. Tests the capacity-budget substitution hypothesis: smaller model should saturate at higher B than 7B. Going down in size (rather than up to 14B) makes this a cleaner positive prediction and avoids sm_75 compatibility risk on the Quadro RTX 6000.

7. *Prompt-level baseline as a difficulty knob (week 4).* Four prompt levels ("Play at beginner / intermediate / expert level", "Play optimally"), no CoT. N=30 per level on Reversi. We do not assume these prompts produce the difficulty levels their labels suggest. Instead, we *measure* the win rate produced by each prompt and report results by measured win rate, not by nominal label. This is itself a test of prompt-based control: if "expert" and "optimal" produce indistinguishable win rates, that is evidence prompts cannot reliably target a specific difficulty. Both budget control and prompt control require per-game empirical calibration; the comparison tests structural properties (continuity, granularity, predictable shape) rather than calibration overhead.

8. *Structured vs. free CoT ablation (week 4).* At B=256 on Reversi, compare two prompt structures: free CoT ("Think through your move.") and structured CoT ("First, identify threats. Second, list 2-3 candidate moves. Third, evaluate each. Fourth, select the best move."). N=30 per condition. Tests whether prompt structure matters as much as budget for reasoning quality. Statistical tests: Fisher's exact test on win rate (binomial outcome), two-sample t-test on per-game move quality (continuous outcome). Reporting both because they may diverge.

9. *Reasoning content analysis (weeks 5-6).* Hand-label ~60 reasoning traces sampled in stratified fashion across 4 budget levels × 3 game phases (early: turns 1-10, mid: middle third, late: last 10 turns), with 5 traces per cell. Sampled only from games where the LLM had ≥3 legal moves (avoiding forced/trivial decisions). Six-category taxonomy adapted from reasoning-trace literature (Mialon et al. 2023, Sprague et al. 2024, Topsakal et al. 2024): state assessment, move enumeration, forward search, opponent modeling, heuristic invocation, verification. Plus quantitative measures: max lookahead depth (integer) and number of candidate moves explicitly considered (integer). Labeling tool will be a 30-minute Streamlit interface backed by SQLite. Pre-coding pass on 10 traces to refine taxonomy before the main labeling run. Analysis: stacked bar chart of category prevalence by budget; violin plot of lookahead depth by budget; logistic regression of P(win) ~ category_flags + budget to test whether specific reasoning operations predict winning independently of budget.

10. *Adaptive difficulty controller on Reversi (week 5).* Sliding-window controller (K=10): if observed win rate over window > 60%, decrease B by one level; if < 40%, increase. Discrete B grid expanded to 8 levels {0, 32, 64, 128, 256, 512, 1024, 2048} for finer-grained tracking. Two opponents (UCT-500 weak, UCT-2000 strong), 200 games each. Comparison baseline: same control logic adjusting between empirically-measured prompt levels (using win rates from Task 7) instead of B. Metrics: convergence time, tracking error, post-convergence variance of B. Cross-validation: the controller's converged B against UCT-2000 should match the B value that yields 50% win rate in Task 3's static curve.

11. *Stochastic decoding robustness check (week 5).* T=0.7 at B ∈ {64, 1024} on Reversi, N=30 each. Tests whether the curve shape survives non-greedy sampling.

12. *Per-phase decomposition (week 5, free re-analysis).* Re-analyze logs from Tasks 3 and 5, stratified by game phase (first 10 turns / mid / last 10 turns). 2D heatmap: x=game phase, y=CoT budget, color=mean move quality. Tests whether budget effects and memory effects concentrate in particular game phases.

13. *Avalon preliminary experiment (week 6, conditional).* If Reversi pipeline is running cleanly by end of week 5, run a budget sweep on Avalon using AvalonBench infrastructure (https://github.com/jonathanmli/Avalon-LLM, built on AgentBench). Use their `avalon-dev-single` config: one LLM agent against four rule-based naive bots. This isolates a single LLM under test against fixed baselines, avoiding multi-agent confounds entirely. Two budget levels (B=64, B=1024), N=20 per condition, fixed role assignment. Verifies the methodology produces a coherent signal in a game where UCT does not apply. *If Reversi is not clean by end of week 5, this task is descoped to a written-only generalization plan in the report.*

14. *Writeup, figures, and final report (week 6).* Twelve figures planned (see Expected Deliverables below).

**Starter code and dependencies.** I am not starting from a research codebase. I will use:
- *Ludii* (https://ludii.games), an existing Java game-playing system with a Python bridge — used as the game engine and UCT baseline.
- *SGLang* (https://github.com/sgl-project/sglang), an existing inference server — used for batched LLM serving with structured output.
- *AvalonBench* (Light et al., NeurIPS 2023 GamesAndAI workshop) — for the preliminary Avalon experiment, I will extend their published agent harness rather than build from scratch.

The work I am doing is the experimental harness that connects these (game serializers, two-pass agent design, memory system, controller, confound-control conditions, and the analysis pipeline), plus the experiments themselves.

**Nice-to-haves if I finish ahead of schedule.** A behavioral diversity analysis (move entropy across games at fixed B, opening-tree diversity) using existing trajectory logs — no new runs needed.

**Team responsibilities.** I am working solo; I am responsible for all components.

---

## Expected Deliverables and Evaluation

The class deliverable is a final report containing the following figures, each tied to a specific question.

**Primary results.**

- *Figure 1.* Win rate vs. CoT budget on Reversi, R1-Distill-7B, with 95% bootstrap CIs. Overlay: prompt-level baseline as four reference points, arithmetic filler condition. The headline figure.
- *Figure 2.* Same plot for Llama 3.2 3B (non-reasoning model). Tests whether the budget effect requires reasoning training.
- *Figure 3.* Adaptive controller time series — B and observed win rate over game index for both UCT-500 and UCT-2000 opponents, with the 50% target line. Shows convergence and tracking.

**Confound isolation.**

- *Figure 4.* Arithmetic filler vs. CoT vs. zero-budget at matched token counts. If filler win rate sits near the B=0 baseline, reasoning is isolated as the mechanism.
- *Figure 5.* Memorization probe — distribution of move entropy across 50 uncertain positions at T=0.7, B=0. High entropy = genuine uncertainty; low entropy = memorization signature.
- *Figure 6.* Constrained vs. unconstrained decoding win rate at B=256.

**Mechanism characterization.**

- *Figure 7.* Memory design comparison — full history vs. structured summary win rate, with per-phase breakdown.
- *Figure 8.* Per-phase × budget heatmap of move quality.
- *Figure 9.* T=0 vs. T=0.7 budget curves on Reversi (decoding robustness).
- *Figure 10.* Reasoning content composition — stacked bar chart of category prevalence (state assessment, move enumeration, forward search, opponent modeling, heuristic invocation, verification) by budget level. Tests whether higher B adds new reasoning operations or just produces more of the same.
- *Figure 11.* Lookahead depth distribution by budget (violin plot) and logistic regression coefficients showing which reasoning operations predict winning independently of budget (bar chart with CIs).
- *Figure 12.* Structured vs. free CoT comparison at B=256 — side-by-side win rate (Fisher's exact) and per-game move quality (t-test).

**Generalization.**

- *Figure 13.* Capacity-budget substitution — Reversi budget curves for 1.5B vs. 7B. Tests whether smaller models saturate at higher B.
- *Figure 14.* Avalon preliminary budget sweep at B=64 vs. B=1024 (if Task 13 runs) or planned-experiment description (if descoped).

**Definition of success.** The project succeeds if the following hold:

1. W(B) is monotonically increasing on Reversi for at least one model with statistically distinguishable adjacent budget levels before saturation (Figure 1).
2. Filler win rate matches B=0 baseline rather than full CoT, isolating reasoning as the mechanism (Figure 4).
3. The memorization probe shows mean entropy > 0.5 on uncertain positions, ruling out trivial pattern-matching (Figure 5).
4. The adaptive controller converges to within ±10pp of the 50% target win rate against both UCT-500 and UCT-2000 (Figure 3).
5. The controller's converged B values are consistent with the static curve from Figure 1 (cross-validation).
6. Reasoning content analysis yields interpretable category distributions across budget levels (Figure 10), with at least one quantitative measure (lookahead depth or candidate count) showing a statistically significant trend with budget (Figure 11).

Null results on any of these are valid contributions if failure modes are characterized. A non-monotonic curve, a filler condition that matches full CoT (suggesting length artifacts dominate), or a controller that fails to converge would each be reportable findings if explained.

---

## Biggest Risks

**Risk 1 — JPype under async load.** JPype has known issues with multi-threaded async Python, which is exactly what batched self-play needs. *Mitigation: test concurrency in week 1 before committing the rest of the pipeline. Fall back to subprocess-based Ludii invocation or pre-computed legal-move caches if JPype fails. This is the single biggest schedule risk and the first thing I will validate.*

**Risk 2 — sm_75 compatibility on the Quadro RTX 6000.** Some SGLang/CUDA kernels have degraded performance on sm_75. *Mitigation: test in week 1; fall back to llama.cpp GGUF for the secondary model if SGLang is incompatible.*

**Risk 3 — R1-Distill at B=0 may be out-of-distribution.** R1-Distill is trained to always produce reasoning tokens; forcing B=0 may collapse output quality. *Mitigation: test explicitly in week 2; if collapse occurs, report curves starting from B=64 and note this as a limitation.*

**Risk 4 — Throughput estimate may be optimistic.** Two-pass generation plus a per-turn summarizer call is effectively three forward passes per decision. Real throughput may be 2–3× lower than the single-pass estimate. *Mitigation: profile early in week 2; if throughput is too low, descope memory comparison from N=30 to N=20 and reduce model size comparison to one game.*

**Risk 5 — Summarizer self-confound.** The summarizer is the same model under test, so its quality may correlate with budget. *Mitigation: summary stability check in Task 5 (BLEU/embedding similarity across budget conditions). The full diagnostic (oracle summary from a larger model) is descoped to summer.*

**Risk 6 — Avalon descoping cascade.** If Reversi runs into trouble in weeks 1–5, Task 13 must be cut. *Mitigation: explicit branch point at end of week 5. If Reversi is not producing clean curves by then, Avalon becomes a written-only plan in the report. The Reversi-only result is independently complete.*

**Risk 7 — Controller oscillation around plateaus.** A bandit-style controller can oscillate if the budget-difficulty curve has flat regions. *Mitigation: K=10 sliding window provides smoothing; report post-convergence B variance as an explicit metric so oscillation is observable rather than hidden.*

---

## What I Need Help With

- *Methodological feedback:* whether the confound control design (arithmetic filler + memorization probe on uncertain positions + memory ablation) is sufficient to isolate reasoning as the mechanism at workshop publication bar.
- *Scoping confirmation:* whether the Reversi-deep + Avalon-preliminary structure is appropriate, given that the project intentionally focuses on depth in one game with a generalization probe rather than breadth across multiple games.
- *Avalon harness:* references or contacts for AvalonBench if available — the preliminary experiment depends on extending their published infrastructure rather than building from scratch.
- *Computing resources:* the RTX 5080 + Quadro RTX 6000 setup should be sufficient. Flagging in case GPU cluster access would let me increase N or add a third model size comparison.

---

## References

- Brown et al. "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling." 2024.
- Wu et al. "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models." 2024.
- Snell et al. "Scaling LLM Test-Time Compute Optimally." ICLR 2025.
- Han et al. "Token-Budget-Aware LLM Reasoning (TALE)." ACL 2025.
- Muennighoff et al. "s1: Simple Test-Time Scaling." 2025.
- Costarelli et al. "GameBench: Evaluating Strategic Reasoning Abilities of LLMs." 2024.
- Light et al. "AvalonBench." NeurIPS 2023 GamesAndAI workshop.
- Gu et al. "LLMs May Not Be Human-Level Players, But They Can Be Testers." CHI 2025.
- Côté et al. "TextWorld: A Learning Environment for Text-based Games." IJCAI 2019.
- Piette et al. "Ludii — The Ludii General Game System." ECAI 2020.
- Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024.
- Dong et al. "XGrammar: Flexible and Efficient Structured Generation Engine for LLMs." MLSys 2025.
- Liu et al. "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024.
- Mizrahi et al. "State of What Art? A Call for Multi-Prompt LLM Evaluation." TACL 2024.
- Boubdir et al. "Elo Uncovered: Robustness and Best Practices in Language Model Evaluation." 2023.
- Hu et al. "A Survey on Large Language Model-Based Game Agents." arXiv 2404.02039.
- Mialon et al. "Augmented Language Models: A Survey." 2023. — reasoning operation taxonomy
- Sprague et al. "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning." 2024. — what CoT does on different task types
- Madaan et al. "Self-Refine." 2024. — categorizes refinement operations
- Topsakal et al. "Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions." 2024. — labeling scheme for reasoning operations
- Wang et al. "Chain-of-Thought Reasoning Without Prompting." 2024. — CoT mechanism analysis
