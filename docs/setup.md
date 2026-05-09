# Setup & Quick Start

This page is the developer onboarding for the CoT-difficulty-knob harness. The
README focuses on the research narrative; everything you need to run the code
lives here.

For Linux / RTX-5080 production hardware, there is a separate, more detailed
guide in [`pc_dev_setup.md`](pc_dev_setup.md). This page is the Mac dev path.

---

## 0. One-shot Mac dev setup

```bash
bash scripts/bootstrap_mac.sh
```

Installs (idempotently): OpenJDK 17, Ollama, the `deepseek-r1:7b` model
(DeepSeek-R1-Distill-Qwen-7B Q4_K_M, ~4.7 GB), Ludii.jar v1.3.14, and all
Python deps (uv-managed, Python 3.11).

The bootstrap is idempotent — running it on a half-installed machine is safe.

---

## 1. Validate the harness with the deterministic mock backend

```bash
uv run pytest -q                                    # 26 tests, ~45s
uv run python scripts/run_budget_sweep.py configs/smoke_n3_mock.yaml
uv run python scripts/analyze_run.py <run_id>       # prints summary + PNGs
```

This proves the full pipeline (config → sweep → SQLite → JSONL → plots) works
without any LLM serving. The mock LLM is deterministic and always loses to UCT
— that's expected; the curve at every B is 0% win rate. See the entry for
`run_3b5e0e753407` in [`run_log.md`](run_log.md) for the canonical reference
result on this trivial baseline.

---

## 2. Run the real Ollama smoke sweep

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

---

## 3. Single-game sanity check

```bash
uv run python scripts/run_smoke_game.py --backend mock --budget 64
uv run python scripts/run_smoke_game.py --backend ollama --budget 256
```

---

## 4. Linux / RTX-5080 production setup

If you're bringing the harness up on the production box, see
[`pc_dev_setup.md`](pc_dev_setup.md). It covers the SGLang vs vLLM choice on
sm_120 (Blackwell), Ollama-as-fallback, and what to validate before trusting
any sweep.

---

## 5. Where data lands when you run something

- `data/results.db` — normalized SQLite source of truth (runs, trials, turns,
  model_calls, summaries, controller_decisions).
- `data/runs/<run_id>/*.jsonl` — append-only per-trial event log (raw backup;
  survives DB corruption).
- `data/runs/<run_id>/figures/*.png` — analysis plots from `analyze_run.py`.

Schema reference: [`tracking_schema.md`](tracking_schema.md).
