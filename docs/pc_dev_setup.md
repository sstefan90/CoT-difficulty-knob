# PC Dev Setup — RTX 5080 Blackwell

**Audience:** Self / future me, bringing the repo up on a fresh PC with an RTX 5080.
**Status:** Forward-looking. None of these steps have been executed on the new box yet.
**Companion docs:** [`architecture.md`](architecture.md) (target stack reference, sm_120 caveats), [`nim_test_backlog.md`](nim_test_backlog.md) (what to run first), [`avalon_test_plan.md`](avalon_test_plan.md).

---

## 1. Target hardware and toolchain

| Component | Spec |
|-----------|------|
| GPU | **RTX 5080 (Blackwell, GB203, sm_120)**. 16 GB VRAM. |
| CPU / RAM | Anything reasonable. Llama 3.1 8B Q4_K_M needs ~6 GB VRAM, so the rest is for batching headroom. |
| OS | Linux (preferred — better SGLang/vLLM support) or Windows + WSL2. |
| CUDA driver | 12.6+ (Blackwell support). Match the toolkit your serving stack expects. |
| CUDA toolkit | 12.4 / 12.6 — install only if you are building wheels from source. Most prebuilt wheels are fine. |
| Python | 3.11 (managed by `uv`, matching the Mac dev path). |
| Java | OpenJDK 17 (deferred — only needed when Ludii/JPype is brought online). |
| Git | Any recent. |

> **Compatibility note carried over from [`architecture.md`](architecture.md):** As of early 2026, SGLang has known issues with **INT4 / GPTQ-marlin kernels on sm_120** (sgl-project issues #15043, #20670; PR #17331). Plan for an SGLang → vLLM fallback. Prefer **NVFP4** over **INT4** when available — NVFP4 is the Blackwell-native format.

---

## 2. Repo bring-up

```bash
git clone <repo-url>
cd CoT-difficulty-knob
uv sync                           # creates .venv, installs all deps
uv run pytest -q                  # 26 tests, ~45 s — uses the deterministic mock backend, no GPU required
```

If `pytest` is green you have validated:

- Pydantic config schemas
- SQLite Store
- JSONLWriter
- Mock LLM backend
- Sweep + runner skeleton

That confirms the platform-agnostic spine of the harness works on the new box before any GPU is touched.

---

## 3. Choose a serving stack

You have three options; the recommendation depends on what experiment you are trying to run.

### 3.1 Ollama (recommended first step)

**Pros:**

- Same client code path as Mac dev. `OllamaClient` ([`src/cot_knob/llm/ollama_client.py`](../src/cot_knob/llm/ollama_client.py)) is fully working.
- One-command model pulls; Q4_K_M weights are 4–6 GB.
- No sm_120 quantization pitfalls — Ollama's GGUF runtime works on Blackwell out of the box.
- Lets you reproduce existing Mac runs as a smoke test.

**Cons:**

- Single-stream by default. No async batching. Throughput is GPU-bound but not GPU-saturating.

**Setup:**

```bash
# Linux (or WSL2)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                                  # starts the daemon on :11434

# Pull the models we use
ollama pull llama3.1:8b                         # Llama 3.1 8B Instruct, Q4_K_M, ~4.7 GB
ollama pull deepseek-r1:7b                      # R1-Distill-Qwen-7B, Q4_K_M, ~4.7 GB (R1 comparison only)
```

Confirm GPU usage:

```bash
nvidia-smi
ollama run llama3.1:8b "hello" --verbose       # prints tokens/sec
```

Existing configs (e.g. `configs/nim_n30_free.yaml`) work as-is. Run the smoke test below in §5.

### 3.2 SGLang — **PRIMARY backend (2026-05-09, RTX 5080)**

**Status (2026-05-09):** Fully implemented. `SGLangClient` rewrote from stub to production-ready on 2026-05-09. Uses `/v1/chat/completions` (applies chat template automatically) + regex-constrained decoding (`sampling_params.regex`). All Nim experiments going forward use `backend: sglang`.

**Why SGLang is now primary:**
- Regex-constrained decoding eliminates `parse_failed` as a confound — every LLM turn produces a valid, legal move.
- ~350 tok/s vs 90 tok/s Ollama → 4× faster sweeps.
- `generate_choice()` for Reversi Pass-2 uses regex alternation over legal moves.

**Install:**

```bash
# In the project venv
uv pip install "sglang[all]" flashinfer-python

# Verify CUDA capability (should be (12, 0) for RTX 5080 Blackwell)
uv run python -c "import torch; print(torch.cuda.get_device_capability())"

# Verify FlashInfer
uv run python -c "import flashinfer; print(flashinfer.__version__)"
```

> **sm_120 fallback**: If FlashInfer kernels are not yet compiled for sm_120, add
> `--attention-backend triton --disable-cuda-graph` to the server launch command.

**Launch the server (from a real terminal — not the Cursor sandbox):**

```bash
# VERIFIED (2026-05-10, RTX 5080, WSL2, SGLang 0.5.9):
# FP8 quantization is required — bfloat16/float16 OOMs during weight load on 16 GB VRAM.
# CUDA_HOME must be set explicitly in WSL2.
CUDA_HOME=/usr/local/cuda-13.2 uv run python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 --host 127.0.0.1 \
  --quantization fp8

# Wait for: "Server is ready" in the log (~30-60 s for model load)
```

**Smoke test — connectivity + constrained decoding:**

```bash
# Quick HTTP check
curl -s http://localhost:30000/v1/models | python3 -m json.tool

# Full harness smoke test (6 games, constrained decoding active)
uv run python scripts/run_budget_sweep.py configs/nim_smoke_sglang.yaml
# Expected: 6 games complete, parse_failed=0%, latency ~3-6s/game
```

**SGLang configs:** All experiment configs have `_sglang.yaml` variants in `configs/`. Use those for all new runs. The Ollama variants remain for reference / fallback.

**Constrained decoding hook-up (already done):**
- `SGLangClient.supports_regex = True` → `LLMAgent._choose_nim()` auto-detects this.
- `NimState.legal_move_regex()` builds the per-turn regex from current pile sizes.
- `constrained_decoding: true` is recorded in JSONL `turn.extra` for every turn.
- `finish_reason` flows through correctly — `"stop"` when regex matched, `"length"` if budget ran out before completion.

### 3.3 vLLM (Blackwell-friendly fallback)

**Pros:**

- More mature sm_120 support for INT4 / FP8 dense models.
- Same `LLMClient` interface — drop-in replacement for `SGLangClient` if SGLang stalls on sm_120.

**Cons:**

- No `choices` operator. Constrained decoding has to go through grammar / outlines / logits-processor extensions, which is more work to wire up than SGLang's native primitive.
- Async API is good but slightly different shape than SGLang's.

**Setup outline:**

```bash
uv pip install vllm                             # check Blackwell wheel availability
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --port 30000
```

`VLLMClient` does not exist yet. Adding one is a half-day task: copy `OllamaClient`, swap the HTTP shape to OpenAI-compatible chat completions, point at the vLLM server's `/v1/chat/completions` endpoint.

---

## 4. Decision tree: which stack for which goal

| Goal | Stack | Why |
|------|-------|-----|
| Reproduce existing Nim runs on the new box | **Ollama** | Same client; no quant surprises; smoke-tests the migration. |
| Run the Nim test backlog T1–T6 (Llama 8B, single-stream) | **Ollama** | Each test is at most a few hundred games. Ollama throughput is enough. |
| Scale to N≥200/cell or run cross-budget batches | **SGLang** (or vLLM) | Async batching is required for reasonable wallclock at large N. |
| Reversi two-pass agent with constrained Pass-2 | **SGLang** | `choices` operator is the cleanest constrained-decoding path. |
| Avalon experiments | **SGLang** or **vLLM** | AvalonBench dialogue is long; batching helps. Constrained decoding less critical here than on Reversi. |
| Anything where SGLang is failing on sm_120 quant kernels | **vLLM** | Fallback. |

---

## 5. Smoke-test recipe (run in this order on the new box)

The goal is to validate the harness end-to-end *before* any new experiment is run.

### 5.1 No-GPU sanity checks

```bash
uv sync
uv run pytest -q
uv run python scripts/run_budget_sweep.py configs/smoke_n3_mock.yaml
uv run python scripts/analyze_run.py <run_id>
```

Expected: pytest green, mock sweep produces a SQLite + JSONL pair under `data/`, analyze_run prints a small summary.

### 5.2 Reproduce the no-LLM B=0 anchor

This is the cheapest GPU-touching test (LLM never actually generates — every turn is a budget=0 random fallback). If results match Mac, the runner is wired correctly.

```bash
ollama serve &
uv run python -m cot_knob.experiments.sweep configs/nim_b0_anchor.yaml
```

Expected: 30 games complete in <1 minute; LLM win rate 50–55% (matches Mac empirical anchor).

### 5.3 Reproduce one real LLM cell

Run a single B=1024 cell from the existing `nim_n30_free` config (or write a 5-game variant). Diff the resulting JSONL trial summaries against the Mac run.

```bash
uv run python -m cot_knob.experiments.sweep configs/nim_n30_free.yaml
# or a small N=5 version for a faster smoke test
```

Per-game `winner` outcomes will not match Mac (different RNG paths even with same seed because tokenization and sampling can differ); that's expected. **What should match within sampling noise:** aggregate win rate (5–10 games is too small to verify; use this only as a "nothing crashes" check).

### 5.4 Calibrate throughput

Measure single-stream Llama 8B Q4_K_M tok/s on the 5080:

```bash
ollama run llama3.1:8b "Generate exactly 1024 tokens of arbitrary text." --verbose
```

Plug the number into [`nim_test_backlog.md`](nim_test_backlog.md) §6 cost-estimate table.

---

## 6. Models to pull

| Model | Source | Quant | Size | Used by |
|-------|--------|-------|------|---------|
| Llama 3.1 8B Instruct | `ollama pull llama3.1:8b` | Q4_K_M (Ollama default) | ~4.7 GB | Current Nim pilot, future Avalon, future Reversi-Llama OOD probe |
| DeepSeek-R1-Distill-Qwen-7B | `ollama pull deepseek-r1:7b` | Q4_K_M | ~4.7 GB | R1 comparison runs (reproducibility of original proposal experiments) |
| Llama 3.1 8B Instruct (FP16) | `huggingface.co/meta-llama/Llama-3.1-8B-Instruct` | FP16 | ~16 GB | Only if SGLang/vLLM tests reveal a quant artifact and we want to compare. |

For SGLang/vLLM, model paths can be HF repo IDs directly; the server will download on first launch.

---

## 7. GPU monitoring

```bash
watch -n 1 nvidia-smi              # one-second refresh, basic
nvtop                              # interactive, more readable; install via apt or build from source
```

Expected behaviour during a Nim sweep:

- VRAM pinned at ~6 GB (Q4_K_M weights + KV cache).
- GPU util oscillates with token generation — single-stream Ollama spends time on prefill.
- During SGLang/vLLM batched runs, VRAM creeps up to 12–14 GB depending on `max_running_requests`.

---

## 8. Deferred but flagged

These are needed for the proposal's *target* Reversi/Ludii pipeline but are not on the critical path for Nim or Avalon work:

- **OpenJDK 17** (`apt install openjdk-17-jdk-headless`).
- **Ludii.jar** (download from [ludii.games](https://ludii.games), drop into `third_party/`).
- **JPype** (`uv pip install jpype1`).
- Test concurrency-under-async is the proposal's biggest schedule risk (Risk 1 in [`proposal.md`](proposal.md)).

Bring this up only when you actively start Reversi sweeps. Not needed for any Nim or Avalon work.

---

## 9. Configuration changes when moving from Mac → PC

For most existing Nim configs, **no change is needed** — they specify `backend: ollama, name: llama3.1:8b`, which works identically on Mac and PC.

If you switch to SGLang or vLLM, you'll need a new backend type and one extra config field for the server URL:

```yaml
model:
  backend: sglang        # or vllm
  base_url: http://localhost:30000
  name: meta-llama/Llama-3.1-8B-Instruct
  think: false
  temperature: 0.0
```

This requires (a) `LLMConfig` to accept the new backend literal, (b) `build_client` in [`src/cot_knob/llm/factory.py`](../src/cot_knob/llm/factory.py) to route to the right client, (c) the `SGLangClient` stub to be filled in (see §3.2 checklist).

---

## 10. Common gotchas

- **NVIDIA driver mismatch:** Blackwell requires recent drivers. Check `nvidia-smi` reports CUDA 12.6+ before installing CUDA-pinned wheels.
- **WSL2 GPU passthrough:** Works for CUDA, but Ollama daemon under WSL2 sometimes needs explicit `OLLAMA_HOST=0.0.0.0` to be reachable from native Windows clients. Native Linux is simpler.
- **`uv` vs `pip` quirks:** `uv pip install` for ad-hoc things (`vllm`, `sglang[all]`), `uv sync` for project deps. Don't mix global `pip install` into the project venv.
- **First Ollama run after `pull` is slow:** The model file has to be loaded to VRAM. Subsequent runs are fast.

---

## 11. Migration checklist (single page)

- [x] Clone repo + `uv sync` + `uv run pytest -q` — **30/30 pass** (2026-05-08)
- [x] T1–T6 YAML configs created in `configs/nim_t*.yaml` (2026-05-08)
- [x] `scripts/bootstrap_linux.sh` fleshed out (2026-05-08)
- [x] Ollama binary installed (`/usr/local/bin/ollama` v0.23.2) (2026-05-09)
- [ ] **Start Ollama daemon** and pull models — run from a real terminal (GPU must be visible):
      ```bash
      ollama serve &                   # start daemon; check /tmp/ollama.log if it fails
      ollama pull llama3.1:8b          # ~4.7 GB
      ollama pull deepseek-r1:7b       # ~4.7 GB (optional for now; needed for R1 comparison)
      ```
      Or run the full bootstrap: `bash scripts/bootstrap_linux.sh`
- [x] Calibrate single-stream tok/s — **measured: ~90 tok/s** (eval rate 88–97 tok/s, 2026-05-09)
      `nim_test_backlog.md §6` updated with calibrated estimates. Model cold-load is ~28s (one-time).
      To re-measure: `ollama run llama3.1:8b "Count from 1 to 500, one number per line." --verbose 2>&1 | grep "eval rate"`
      To re-estimate: `uv run python scripts/estimate_sweep_time.py configs/nim_t1_step_budget_n85.yaml --tps 90`
- [ ] Run `nim_b0_anchor` config — expect ~50% win rate, <2 min wallclock:
      ```bash
      uv run python scripts/run_budget_sweep.py configs/nim_b0_anchor.yaml
      ```
- [ ] Run T1 (255 games, ~1h 14m at 200 tok/s):
      ```bash
      uv run python scripts/run_budget_sweep.py configs/nim_t1_step_budget_n85.yaml
      ```
- [ ] Run T2 (170 games each, ~40m at 200 tok/s — can run in sequence or parallel):
      ```bash
      uv run python scripts/run_budget_sweep.py configs/nim_t2_step_n85.yaml
      uv run python scripts/run_budget_sweep.py configs/nim_t2_free_n85.yaml
      ```
- [ ] **STOP-AND-REASSESS** after T1+T2. Analyze:
      ```bash
      uv run python scripts/analyze_run.py <run_id>
      ```
- [ ] Decide: Ollama only for Nim T3–T6, or stand up SGLang/vLLM first?
- [ ] If SGLang: confirm sm_120 quant works; flesh out `SGLangClient`
- [ ] If vLLM: write `VLLMClient`
- [ ] (Deferred) JDK 17 + Ludii.jar + JPype, only when Reversi work begins
