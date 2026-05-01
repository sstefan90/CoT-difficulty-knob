# SQLite tracking schema

The canonical schema lives at
[`src/cot_knob/tracking/schema.sql`](../src/cot_knob/tracking/schema.sql).
This document explains *why* each table exists and how the analyses in
`docs/proposal.md` map onto it.

## Tables

### `runs`
One row per `run_budget_sweep.py` invocation. Captures provenance:
git SHA, config hash, host, GPU, status, start/end timestamps, and the
verbatim YAML config.

### `trials`
One row per game played. `condition_json` carries the full experimental
cell (model, B, opponent, memory mode, prompt variant, seed). `winner`
is `llm`, `uct`, or `draw`.

### `turns`
One row per LLM decision. Records the board state, legal moves, the
chosen move, and the UCT top-3 at that state for move-quality analysis
(used in proposal Tasks 3, 8, 12). `phase` is `early`/`mid`/`late` so
the per-phase decomposition (Task 12) is a `GROUP BY` away.

### `model_calls`
One row per HTTP request to the LLM backend. `role` is one of:

- `reason` — Pass 1, free CoT (this is the reasoning corpus Task 9 will label)
- `select` — Pass 2, constrained move selection
- `summarize` — memory summarizer
- `filler` — arithmetic filler control (Task 4)

### `summaries`
One row per memory state, written *after* the summarizer runs each turn.
Used for the Task 5 summary-stability sub-analysis.

### `controller_decisions`
Empty until proposal Task 10 (adaptive controller). Reserved here so the
schema doesn't change later.

## Analyses → SQL queries

| Proposal section                            | Query shape                                                       |
| ------------------------------------------- | ----------------------------------------------------------------- |
| Fig 1. Win rate vs B                        | `SELECT B, AVG(winner='llm') FROM trials JOIN ... GROUP BY B`     |
| Fig 4. Filler vs CoT vs B=0                 | Same, partitioned by `condition_json->>'agent_kind'`              |
| Fig 8. Per-phase × budget heatmap           | `SELECT phase, B, AVG(move_quality) FROM turns ... GROUP BY ...`  |
| Task 9 reasoning labeling corpus            | `SELECT response_text FROM model_calls WHERE role='reason'`       |
| Task 5 summary stability                    | `SELECT summary_text FROM summaries WHERE after_turn_idx=...`     |
