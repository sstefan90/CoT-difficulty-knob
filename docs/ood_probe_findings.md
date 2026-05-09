# OOD Probe Findings — Reversi (DeepSeek R1 7B)

> **Scope.** This file is the **Reversi** OOD probe story (R1-Distill-7B legality / move-identification). The **Nim** OOD probe (Llama 3.1 8B) lives in [`docs/nim_experiment_findings.md`](nim_experiment_findings.md) §3. The two games stress different parts of the model — Reversi exposes board-parsing weakness; Nim exposes arithmetic-execution weakness — and should not be conflated.
>
> **Why we run OOD probes at all.** Before we trust a win-rate-vs-budget curve, we need to know that the LLM can *participate* in the game without trivial scaffolding. If the agent cannot identify legal moves from the board representation, the entire sweep collapses to "did the regex extract anything," not "did reasoning improve play."

**Script:** `scripts/probe_ood.py`  
**Output:** `data/probes/ood_raw_legality.jsonl`

---

## Probe versions

### v1 — Pass-2 constrained choice (INVALIDATED)

> **Why we ran this.** First attempt at measuring "can the model pick a reasonable move." Used the existing two-pass agent (Pass-1 reasons, Pass-2 picks from a list of legal moves) directly as the probe.
>
> **What we learned.** The probe was load-bearing on Pass-2's *fallback to `choices[0]`* — even total reasoning failure produces 100% legal output, because the harness picks for the model. The 100% legality headline was a measurement artifact, not a capability finding. Only the 20% UCT-top1 number — which depends on the model's actual choice — was real, and it was barely above random.

**Date:** 2026-05-04  
**Config:** `--n-positions 15 --uct-game-iters 50 --uct-eval-iters 100 --no-t07`  
**Output:** `data/probes/ood_t0_probe.jsonl`

**What it did:** The old probe used `generate_choice` (Pass-2) which hands the model
a list of all legal moves and has a fallback to `choices[0]`. This trivially guarantees
100% legal output regardless of the model's actual understanding of the position.

**Why it was invalid:**

1. **100% legal is guaranteed by construction.** Pass-2 prompt says:
   `"Choose exactly one of: d3, c4, f5, e6"`. The model echoes back one token.
   Even on total failure, `_match_choice` falls back to `choices[0]`.

2. **`finish_reason=length` is benign noise.** `num_predict=16` always triggers a
   length finish; it says nothing about reasoning quality.

3. **20% UCT-top1 was the only real signal.** Without CoT, the model picked the
   best UCT move 3/15 times — only slightly above the random baseline (~12–14%).
   This number is valid; the 100% legal headline was not.

**Discarded results:**

| Metric | Value | Valid? |
|---|---|---|
| Legal moves | 100% (15/15) | ❌ Artifact |
| UCT top-1 picks | 20% (3/15) | ✓ Real signal |
| Collapses | 0 | ❌ Artifact |

---

### v2 — Raw legality sweep (CURRENT, STRONGER)

> **Why we ran this.** v1 was invalidated by the Pass-2 fallback. We needed a probe that fails honestly when the model produces nothing parseable: no legal-move list in the prompt, no constrained-choice fallback, raw coordinate extraction only. This is also a budget sweep so we can see whether more CoT budget recovers legality.
>
> **What we learned.** The legality curve is essentially flat at zero. Even at B=512 the model produces a legal Reversi move on only 1/15 positions; with no scaffolding it cannot derive legality from the ASCII board. It instead returns canonical opening squares (`d5`, `e5`, `g6`, …) regardless of game state — pattern-matching pretraining priors, not board computation. **Implication for the main sweep:** the legal-move list in Pass-2 is not a "convenience" — it is what makes the agent functional at all.

**Date:** 2026-05-04  
**Config:** `--n-positions 15 --raw-budgets 0 64 256 512 --uct-game-iters 50 --uct-eval-iters 100 --no-t07`  
**Output:** `data/probes/ood_raw_legality.jsonl`

**What changed:**

| Dimension | v1 (invalid) | v2 (current) |
|---|---|---|
| Legal moves listed? | ✓ explicit in prompt | ✗ omitted (`show_legal=False`) |
| Pass-2 fallback? | ✓ always picks legal | ✗ none — raw extraction only |
| Move extraction | constrained choice | last `[a-h][1-8]` in response |
| Response budget | 16 tokens | 512 tokens |
| Budgets tested | B=0 only | B ∈ {0, 64, 256, 512} |

**Query modes:**

- **B=0 (no CoT):** Single call, `think=False`, `num_predict=512`. Model must answer
  purely from pattern recognition — no explicit chain of thought.
- **B>0 (with CoT):** Pass-1 (`think=True`, `num_predict=B`) captures reasoning. Pass-2
  prepends the reasoning and asks for a final coordinate with `num_predict=512`.
- **Extraction:** Last `[a-h][1-8]` found in the response text. The model typically
  deliberates ("d3 is risky, e5 is better") and concludes with its final pick.

**Why this test is meaningful:**

The model must:
1. Read the ASCII board (orientation header, grid, piece lists).
2. Calculate which squares would flip opponent pieces (Reversi legality rule).
3. Name a coordinate that satisfies this condition — entirely on its own.

This directly measures **board-parsing and move-identification capability** without
any scaffolding. A coordinate-only regex search with no fallback means every illegal
pick counts against the model.

---

## v2 Results (COMPLETE)

**Run completed:** 2026-05-04T05:52:54Z — 60 queries, 15 positions × 4 budgets.  
**Total runtime:** ~31 minutes.

```
  Budget     Legal   UCT-top1   UCT-top3   Collapse
  ──────────────────────────────────────────────────
  B=0         0.0%      0.0%      0.0%      0.0%  (0/15)
  B=64        0.0%      0.0%      0.0%      0.0%  (0/15)
  B=256       0.0%      0.0%      0.0%      0.0%  (0/15)
  B=512       6.7%      0.0%      0.0%      0.0%  (1/15)
```

**The legality curve is completely flat.** More thinking budget provides essentially
zero improvement in the model's ability to identify legal moves without being given
the list.

### The `d5` signal

Across 60 queries the model outputs `d5` **11 times** — across early, mid, and
late-game positions, including positions where d5 has been occupied for 40+ turns.
`e5`, `g6`, `e4`, `c5` are also repeatedly chosen. These are all canonical Reversi
opening moves — the model has a strong prior from pre-training that these are
"good Reversi squares" and returns to them regardless of actual board state.

**Confidence is not the issue**: Collapse rate is 0% at every budget. The model
never fails to produce a coordinate — it produces the wrong one *confidently*,
without verifying legality.

---

## Interpretation

> **Why this matters for the project.** The OOD probe is the load-bearing sanity check before any Reversi sweep is interpretable. v2's flat legality curve says the headline `W(B)` curve must be read as *strategic discrimination given a pre-validated legal set*, not "how well does the LLM play Reversi end-to-end." It also explains why R1 + Reversi was a worse fit than Llama + Nim for the pilot: the Reversi bottleneck (board parsing) is **upstream** of the CoT knob.

The model is doing **pattern-matching**, not board-state computation:

- It knows `d5` is a canonical Reversi move → outputs `d5`
- It does NOT traverse the 8 directions from an empty square to verify flips
- More CoT budget (up to B=512) barely helps (1/15 = 6.7%) because the bottleneck
  is spatial computation from an ASCII grid, not strategic reasoning depth

**Core finding:** R1-Distill-7B cannot derive Reversi legal moves from the board
representation alone. The Pass-2 legal-moves scaffold is **load-bearing**, not a
convenience.

**Implication for the budget sweep:** The difficulty knob does not control
board-reading ability — it controls *strategic discrimination among a pre-validated
list of legal moves*. The paper framing should reflect this: CoT budget → quality
of move selection, given the legal set is provided.

---

## Open items

- [x] Populate v2 results table
- [ ] Run T=0.7 memorization probe (expect very low entropy — model will
      consistently pick `d5`/`e5` heuristics regardless of uncertain positions)
- [ ] Extend to higher budgets (B=1024, B=2048) to confirm legality curve stays flat
- [ ] After main budget sweep, use `oracle_chosen_rank / n_legal_moves` (percentile
      rank) as primary metric — it's unaffected by this OOD finding since Pass-2
      handles legality
