# OOD Probe Findings

**Script:** `scripts/probe_ood.py`  
**Output:** `data/probes/ood_raw_legality.jsonl`

---

## Probe versions

### v1 — Pass-2 constrained choice (INVALIDATED)

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

## v2 Results (pending)

The full sweep is currently running. Expected completion: ~70 minutes.
Results will be populated below once the JSONL is available.

```
  Budget     Legal   UCT-top1   UCT-top3   Collapse
  ──────────────────────────────────────────────────
  B=0        TBD       TBD        TBD        TBD
  B=64       TBD       TBD        TBD        TBD
  B=256      TBD       TBD        TBD        TBD
  B=512      TBD       TBD        TBD        TBD
```

### Sanity check results (1 position)

Before the full sweep, a 1-position sanity check confirmed extraction is working:

| Budget | Extracted | Legal | Raw output (first 100 chars) |
|---|---|---|---|
| B=0 | `d5` | ✗ | `"To make a legal move ... I recommend moving **d5**..."` |
| B=64 | `h1` | ✗ | `"To determine the best move for White ... Here's a step-by-step breakdown..."` |

**Key observation:** The model produces plausible-sounding coordinates (`d5`, `h1`)
but they are illegal. This is the OOD signal: without the legal-moves list, the
model identifies coordinates on the board but cannot reliably verify which ones
satisfy the flip condition.

---

## Interpretation guide (once results arrive)

| Legal% at B=0 | Verdict |
|---|---|
| < 20% | Strong OOD collapse. Model cannot parse legal moves from board alone. Start curves at B=64. |
| 20–50% | Partial collapse. Model has some board-reading ability but is unreliable. Note as limitation. |
| ≥ 50% | No collapse. Model can identify legal moves without CoT scaffolding. |

The legality-vs-B curve itself is the key deliverable: it shows how much CoT budget
is needed before the model reliably plays legal moves, independent of move quality.

---

## Open items

- [ ] Populate v2 results table once the sweep finishes.
- [ ] After main budget sweep, overlay v2 B=0 legality/top1 data as baseline on win-rate-vs-B curve.
- [ ] Run T=0.7 memorization probe once uncertain positions are identified from the main sweep.
- [ ] Consider testing higher budgets (B=1024, B=2048) in the raw probe to see if legality saturates.
