# CoT Budget as a Compute-Grounded Difficulty Knob for LLM Game Agents

**Stephone Christian** — stephone@stanford.edu  
CS 348K — Visual Computing Systems, Spring 2026

The full project proposal lives at [`docs/proposal.md`](docs/proposal.md).
Architecture / build notes are in [`docs/architecture.md`](docs/architecture.md).
Per-run journal is in [`docs/run_log.md`](docs/run_log.md).

---

## Quick start

### 0. One-shot Mac dev setup

```bash
bash scripts/bootstrap_mac.sh
```

Installs (idempotently): OpenJDK 17, Ollama, the `deepseek-r1:7b` model
(DeepSeek-R1-Distill-Qwen-7B Q4_K_M, ~4.7 GB), Ludii.jar v1.3.14, and
all Python deps (uv-managed, Python 3.11).

### 1. Validate the harness with the deterministic mock backend

```bash
uv run pytest -q                                    # 26 tests, ~45s
uv run python scripts/run_budget_sweep.py configs/smoke_n3_mock.yaml
uv run python scripts/analyze_run.py <run_id>       # prints summary + PNGs
```

This proves the full pipeline (config → sweep → SQLite → JSONL → plots)
works without any LLM serving. It always loses to UCT — that's expected;
the mock is a deterministic stand-in.

### 2. Run the real Ollama smoke sweep

```bash
ollama serve &                                      # if not already running
uv run python scripts/run_budget_sweep.py configs/smoke_n3.yaml
```

Four budgets `B ∈ {0, 64, 256, 1024}` × N=3 seeds = **12 games** against
UCT-2000. Expected wallclock on Apple Silicon with R1-Distill-Qwen-7B
(Q4_K_M): **roughly 60–90 minutes**, dominated by the B=1024 cells.

```bash
uv run python scripts/analyze_run.py <run_id>       # win rate vs B + diagnostics
```

### 3. Single-game sanity check

```bash
uv run python scripts/run_smoke_game.py --backend mock --budget 64
uv run python scripts/run_smoke_game.py --backend ollama --budget 256
```

---

## Repository layout

```
src/cot_knob/
├── llm/         # LLMClient ABC + MockClient + OllamaClient + SGLangClient (stub)
├── games/       # Reversi state + GameState protocol
├── agents/      # LLMAgent (two-pass), UCTAgent (Python MCTS)
├── memory/      # FullHistoryMemory, StructuredSummaryMemory
├── prompts/     # Reversi prompt templates (free + structured CoT)
├── experiments/ # config (pydantic), runner (single match), sweep (config-driven)
├── tracking/    # SQLite Store, JSONLWriter, analytics queries
└── analysis/    # plotting helpers
```

The **Mac↔PC seam** is exactly `src/cot_knob/llm/`. Everything else is
platform-agnostic. On the RTX 5090 box we replace `OllamaClient` with
`SGLangClient` (or `VLLMClient` as a fallback for sm_120 INT4 issues —
see [`docs/architecture.md`](docs/architecture.md)) and re-run the same
configs.

## Where results live

- `data/results.db` — normalized SQLite source of truth (runs, trials,
  turns, model_calls, summaries, controller_decisions).
- `data/runs/<run_id>/*.jsonl` — append-only per-trial event log
  (raw backup; survives DB corruption).
- `data/runs/<run_id>/figures/*.png` — analysis plots from
  `analyze_run.py`.

Schema reference: [`docs/tracking_schema.md`](docs/tracking_schema.md).

## What the bootstrap does *not* yet build

Deferred per the proposal task list:

- **Ludii Java bridge**: pure-Python Reversi is the current opponent.
  `LudiiUCTAgent` (JPype-backed) lands when Ludii.jar is in
  `third_party/`; it conforms to the same `Agent` interface.
- **Cross-game async batching** (proposal Task 2)
- **Arithmetic filler agent** (Task 4) — class is reserved
- **Adaptive controller** (Task 10) — schema row reserved
- **Avalon** (Task 13)
