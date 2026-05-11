# CoT Budget as a Compute-Grounded Difficulty Knob for LLM Game Agents

**Stephone Christian** — stephone@stanford.edu  
CS 348K — Visual Computing Systems, Spring 2026

The full project proposal lives at [`docs/proposal.md`](docs/proposal.md).  
Architecture and build notes are at [`docs/architecture.md`](docs/architecture.md).  
Detailed Nim findings (all tables, traces, mechanism analysis) are at [`docs/nim_experiment_findings.md`](docs/nim_experiment_findings.md).

---

## What this project does

The project asks a simple question: if you give a language model more tokens to think before it acts in a game, does it play better? We call the token limit `B` — the **CoT budget** — and treat it as a dial that controls reasoning depth. The game is the oracle: win rate vs `B` gives a clean, automatic measure of whether more thinking helps.

We run the full experiment pipeline on **Nim [3,5,7]** (a solved combinatorial game with an exact optimal strategy) using **Llama 3.1 8B Instruct** served by SGLang. Nim is the pilot: it has a known closed-form solution, so we can measure model performance against both a random baseline (50% theoretical win rate) and a perfect opponent (0% win rate unless the model plays optimally).

The Nim results are complete and serve as a **negative control**. The project moves next to **Avalon**, where there is no closed-form optimal policy and the budget knob is expected to operate differently.

---

## Quick start

### 0. Set up the environment

```bash
# Clone and install Python deps
git clone <repo>
cd CoT-difficulty-knob
uv sync
```

### 1. Start the SGLang server (PC with GPU)

Requires an NVIDIA GPU with ≥16 GB VRAM. FP8 quantization is needed to fit Llama 3.1 8B.

```bash
CUDA_HOME=/usr/local/cuda-13.2 uv run python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 --host 127.0.0.1 --quantization fp8
```

Wait for `"Server is ready"` before running experiments. See [`docs/pc_dev_setup.md`](docs/pc_dev_setup.md) for full setup instructions.

### 2. Validate the harness with the mock backend

```bash
uv run pytest -q                                         # unit tests (~45s)
uv run python scripts/run_budget_sweep.py configs/smoke_n3_mock.yaml
uv run python scripts/analyze_run.py <run_id>
```

This runs the full pipeline (config → sweep → SQLite → JSONL → plots) without any LLM server. The mock agent loses to UCT every game — that is expected.

### 3. Run a Nim sweep

```bash
uv run python scripts/run_budget_sweep.py configs/nim_vs_random_n30_sglang.yaml
uv run python scripts/analyze_run.py <run_id>
```

Each config file specifies the game, budgets, number of seeds per cell, opponent, and prompt variant. See `configs/` for all run configurations used in the paper.

### 4. Open the analysis notebooks

```bash
cd notebooks
uv run jupyter lab
```

- `06_nim_llama_sweep.ipynb` — main budget sweep: win rate vs `B`, token output vs budget, mistake rate on winning positions
- `07_nim_diagnostics.ipynb` — variant comparison, P1/P2 split, phase-stratified analysis, win trace classification, pooled binomial test, vs-optimal trace inspection

---

## Nim results (pilot, complete)

All experiments use Llama 3.1 8B Instruct, SGLang FP8, Nim [3,5,7], regex-constrained decoding. The theoretical random-play anchor is exactly **50.0%** (verified by game-tree recursion and N=100,000 Monte Carlo).

### Win rate vs budget

![Win rate vs budget and vs optimal](docs/figures/fig1_win_rate_vs_budget.png)

**Left:** free_cot vs random opponent, N=30/cell. No budget cell is individually distinguishable from the 50% anchor — N=30 CIs are ±18 pp. **Right:** free_cot vs NimOptimal agent. 0/132 wins at B>0; upper 95% CI bound ~2%.

| Budget B | k/n | Win rate | 95% CI |
|----------|-----|----------|--------|
| 0 (random fallback) | 11/30 | 36.7% | [21.9%, 54.5%] |
| 64 | 17/30 | 56.7% | [39.2%, 72.6%] |
| 256 | 12/30 | 40.0% | [24.6%, 57.7%] |
| 512 | 13/30 | 43.3% | [27.4%, 60.8%] |
| 1024 | 15/30 | 50.0% | [33.2%, 66.8%] |

### Pooled test across all deliberate-play games

Pooling all B>0, vs-random games across four prompt variants and two sample sizes (520 unique games, duplicates excluded):

> **284/520 = 54.6%** &nbsp; 95% CI [50.3%, 58.9%] &nbsp; one-sided p = **0.020**

The model plays slightly but significantly above chance when given any reasoning budget. The effect is small (+4.6 pp) and appears across all variants — no single condition dominates it.

### Variant comparison and high-N replication

![Variant comparison and T1/T2 replication](docs/figures/fig2_variant_and_t1t2.png)

**Left:** N=30 variant comparison at B=1024. `step_by_step` looks promising at 70% but the Wilson CI includes 50% (p=0.074). **Right:** T1 and T2 at N=85 — the budget gradient and variant advantage both collapse. `step_by_step` and `free_cot` return identical 56.5%.

### Variant comparison (N=30, B=1024, vs random)

| Variant | k/n | Win rate | 95% CI | p vs 50% |
|---------|-----|----------|--------|----------|
| `free_cot` | 16/30 | 53.3% | [36.4%, 69.6%] | ns |
| `nim_sum_given` | 15/30 | 50.0% | [33.2%, 66.8%] | ns |
| `step_by_step` | 21/30 | 70.0% | [52.1%, 83.3%] | ns (p=0.074) |
| `few_shot` | 17/30 | 56.7% | [39.2%, 72.6%] | ns |

No variant is individually significant. The `step_by_step` 70% result did not replicate: at N=85 it returned 56.5%, identical to `free_cot` at the same N. The N=30 result was sampling noise.

### High-N confirmation: T1 and T2 (N=85)

| Condition | k/n | Win rate | 95% CI | p vs 50% |
|-----------|-----|----------|--------|----------|
| `step_by_step` B=256 | 46/85 | 54.1% | [43.6%, 64.3%] | ns |
| `step_by_step` B=512 | 50/85 | 58.8% | [48.2%, 68.7%] | ns |
| `step_by_step` B=1024 | 48/85 | 56.5% | [45.9%, 66.5%] | ns |
| `free_cot` B=1024 | 48/85 | 56.5% | [45.9%, 66.5%] | ns |

No budget gradient. No variant advantage. Both T1 and T2 are negative.

### vs optimal opponent (N≈30/cell, all budgets)

| Budget | k/n | Win rate | 95% CI |
|--------|-----|----------|--------|
| 0 | 1/30 | 3.3% | [0.6%, 16.7%] |
| 64–512 | 0/90 | 0.0% | [0.0%, ~4%] |
| 1024 | 0/42 | 0.0% | [0.0%, 8.4%] |

**0/132 wins at B>0.** Upper 95% CI bound ~2%. The model cannot compete against a player that always computes nim-sum correctly.

### Phase-stratified mistake rate and pooled test

![Phase stratified and pooled binomial](docs/figures/fig3_phase_and_pooled.png)

**Left:** Mistake rate on winning-position turns by game phase. The model is *worse than random* in the early game (66%, p<0.001) when pile sizes are largest and XOR arithmetic is hardest, and *essentially random* in the mid-game (50%). **Right:** Pooled binomial across 520 unique deliberate-play games: 54.6% [50.3%, 58.9%], p=0.020 — marginally but significantly above chance.

### Phase-stratified mistake rate

Mistake rate = fraction of winning-position turns where the model chose a losing move. Pooled across all variants and budgets.

| Phase | Mistakes/turns | Mistake rate | 95% CI | p vs 50% |
|-------|----------------|--------------|--------|----------|
| Early (turns 0–3) | 274/415 | **66.0%** | [61.3%, 70.4%] | p<0.001 |
| Mid (turns 4–11) | 177/355 | **49.9%** | [44.7%, 55.0%] | ns |
| Late (turns ≥12) | 0/0 | — | — | — |

The model is **worse than random in the early game** and **essentially random in the mid-game**. Opening turns have the largest pile sizes and the hardest XOR arithmetic; as piles shrink the model drifts toward chance. Games on Nim [3,5,7] never reach the late phase (≤15 stones total).

### Win trace analysis

![Win trace classification](docs/figures/fig4_win_trace.png)

**Left:** 95% of wins involved at least one correct move from a winning position — none were purely opponent-induced. **Right:** The model exploits winning positions at 53% in games it wins vs 21% in games it loses (32 pp gap). Wins correlate with correct play, but 53% is barely above the 50% random baseline.

Of 73 B>0 free_cot wins against a random opponent:

- **0%** had no winning positions (no game was purely opponent-induced)
- **95%** were partially earned (model made at least one correct move from a winning position)
- **5%** were fully earned (all winning-position moves were correct)

Correct-move rate from winning positions: **53%** in wins vs **21%** in losses (32 pp gap). Wins correlate with correct play, but 53% is barely above the 50% random baseline — the model wins when it happens to make a correct mid-game move, not because it has a reliable strategy.

### Mechanism

Trace inspection of 5 B=1024 losses against the optimal opponent identifies the failure mode precisely:

> **Strategy-known, arithmetic-failed.** The model names the nim-sum algorithm, writes binary representations, and frames the goal correctly at every turn. It fails because it consistently misevaluates `a XOR b XOR c` for mid-game pile configurations. For example: `2 ⊕ 5 ⊕ 2 = 7` (correct: 5), `2 ⊕ 1 ⊕ 7 = 6` (correct: 4). Longer CoT at B=1024 produces more reasoning tokens but the arithmetic remains wrong — the model cannot self-verify XOR computations. In one trace, the model hallucinates a game move that never occurred.

---

## Repository layout

```
src/cot_knob/
├── llm/         # LLMClient ABC + MockClient + OllamaClient + SGLangClient
├── games/       # Nim state + Reversi state + GameState protocol
├── agents/      # LLMAgent (two-pass: reason → constrained MOVE tag), UCTAgent
├── memory/      # FullHistoryMemory, LastMoveMemory
├── prompts/     # Nim prompt templates (free_cot, nim_sum_given, step_by_step, few_shot)
├── experiments/ # SweepConfig (pydantic), runner (single match), sweep (config-driven)
├── tracking/    # SQLite Store, JSONLWriter, analytics queries
└── analysis/    # plotting helpers
```

The serving backend lives entirely in `src/cot_knob/llm/`. Everything else is platform-agnostic. Swap `OllamaClient` for `SGLangClient` in configs; the game harness and sweep logic do not change.

---

## Where results live

- `data/results.db` — normalised SQLite source of truth (runs, trials, turns, model_calls)
- `data/runs/<run_id>/*.jsonl` — append-only per-trial event log (raw backup)
- `notebooks/06_nim_llama_sweep.ipynb` — main budget sweep analysis
- `notebooks/07_nim_diagnostics.ipynb` — variant, phase, mechanism, and pooled analyses

Schema reference: [`docs/tracking_schema.md`](docs/tracking_schema.md).

---

## Status and next steps

Nim is complete and serves as a **negative control**: on a task reducible to binary arithmetic, CoT budget and prompt structure add ~5 pp pooled but neither approaches optimal play. The budget knob does not modulate performance when the optimal strategy requires reliable symbolic execution.

The project moves to **Avalon** (see [`docs/avalon_test_plan.md`](docs/avalon_test_plan.md)), where:
- There is no closed-form optimal policy
- Strategy requires social reasoning and belief tracking across players
- CoT budget is expected to modulate performance meaningfully

Items deferred from the Nim phase:
- **Cross-game async batching** (proposal Task 2)
- **Adaptive controller** (Task 10)
- **Reversi** — retained as a secondary game if Avalon results need a second data point
