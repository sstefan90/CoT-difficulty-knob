# Run log

One short paragraph per experimental run. Newest first. Mirror the SQLite
`runs.run_id` so we can cross-reference.

## Template

```
### YYYY-MM-DD — <run_id> — <config name>
- Backend: ollama / sglang / mock
- Model: deepseek-r1:7b
- N per cell: 3
- Wallclock: HH:MM
- Outcome: 1-2 sentences. Win rates by B. Anything weird.
- Followups: ...
```

---

### 2026-05-09 — (pending) — OOD + Diagnostic rerun (SGLang, memory-fix session)

- Backend: sglang
- Model: llama3.1:8b (Q4_K_M)
- N per cell: 30 (diagnostic sweeps), 8 positions × 5 budgets (OOD probe)
- Wallclock: TBD
- Outcome: TBD
- Purpose: Re-establish OOD and diagnostic baselines with (1) the memory fix in runner.py
  and (2) SGLang constrained decoding. Two comparisons:
  - `probe_nim_ood.py --backend sglang` vs. prior Ollama run → isolate parse-failure removal
  - `probe_nim_ood.py --backend sglang --constrained` → measure pure strategy quality
  - `nim_diag_[a-c]_sglang.yaml` + `nim_abl_d_forced_loss_sglang.yaml` (N=30, B=1024)
- Followups: After OOD/diag pass, kick off N=30 full Nim sweeps (nim_n30_*.yaml SGLang configs).

---

### 2026-05-09 — multiple — N=30 full Nim PC sweep (memory-fix session)

- Backend: ollama
- Model: llama3.1:8b (Q4_K_M)
- N per cell: 30

**Memory bug discovered and fixed (2026-05-09)**

Root cause: `runner.py` called `memory.update(rec)` with `tel.state_serialized` (the board
state **before** the agent's move) then applied `state.apply_move()` afterwards. This caused
`LastMoveMemory` to render `"Piles after: …"` using the pre-move pile sizes, contradicting the
(correct) board shown in the same prompt. The LLM reasoned about wrong pile values.

Fix: reversed the order — `state.apply_move()` first, then `memory.update(rec)` with
`state.to_serializable()` (post-move). Verified with `scripts/inspect_prompt.py` and 21 new
regression tests (`tests/test_nim_memory_runner.py`). All 51 tests green after fix.

**Data impact:**
- `nim_b0_anchor` (N=30): **valid** — no LLM generation, memory never read by model.
- All prior Mac runs (N=10 full sweep + N=30 diagnostic): **relationally valid** but
  absolutely confounded. Relative comparisons between variants hold; absolute win rates
  may be suppressed at B≥512 where full reasoning executed on wrong pile values.
- Partial T1 attempt (run_af35a3d31fd4, 129 trials): **scrubbed** from DB + disk.

**Runs queued / running on PC (all use memory-fixed code):**
| Config | run_id | Status |
|--------|--------|--------|
| nim_n30_free | run_b3e44e1b8163 | running |
| nim_n30_scaffold | — | queued |
| nim_n30_nimsum | — | queued |
| nim_n30_fewshot | — | queued |
| nim_vs_random_n30 | — | queued after above |
| nim_vs_optimal_n30 | — | queued after above |
| nim_t1_step_budget_n85 | — | queued after above |
| nim_t2_step_n85 | — | queued after T1 |
| nim_t2_free_n85 | — | queued after T1 |
| nim_selfplay_n30 | — | optional, lower priority |

Expected total wall-clock: ~7–9 h sequential on Ollama/5080 @ 90 tok/s.

- Followups: Analyze nim_n30_* once complete; then T1 → T2 stop-and-reassess point.
  If T1+T2 both negative → move to Avalon. SGLang migration doc written at
  `docs/sglang_migration.md` for when higher throughput is needed.

---

### 2026-05-09 — (no run_id) — RTX 5080 throughput calibration

- Backend: ollama
- Model: llama3.1:8b (Q4_K_M)
- N per cell: n/a (throughput probe only)
- Wallclock: ~30s
- Outcome: Measured **eval rate: 88–97 tok/s** single-stream on RTX 5080 (Blackwell, WSL2).
  Cold-load time ~28s (one-time per daemon session). Using 90 tok/s as planning figure.
  At this rate: T1 (255 games) ~2h 32m; T1+T2 (425 games) ~5h 20m; all T1–T6 ~15h.
  These are model-based estimates — actual times depend on prompt overhead and game length
  variance. `nim_test_backlog.md §6` updated with calibrated table.
- Followups: Run nim_b0_anchor → T1 → T2 (stop and assess). See migration checklist.

---

### 2026-05-09 — (no run_id) — estimate_sweep_time.py fix + cost table update

- Backend: n/a (tooling only)
- Model: n/a
- N per cell: n/a
- Wallclock: ~15 min
- Outcome: Fixed two bugs in `scripts/estimate_sweep_time.py` that caused wildly inflated
  estimates for Nim configs: (1) oracle_iterations=0 was ignored (charged 0.30s/turn anyway),
  (2) non-UCT opponents (random, nim_optimal) used UCT-2000 cost (0.286s/turn) from the
  schema default. Also added auto-detection of avg turns/game from game type
  (nim[3,5,7]→10, nim[7,11,13]→25, reversi→60), dual GPU speedup hints
  (Ollama/5080 ~200 tok/s and SGLang/5080 ~400 tok/s), and corrected the SGLang
  default TPS from 80→400. Updated `nim_test_backlog.md §6` cost table with corrected
  numbers; updated `pc_dev_setup.md §11` migration checklist with Ollama install
  status and per-step commands.
- Followups: Ollama binary is installed but daemon not yet started and models not pulled.
  From a real terminal: `ollama serve &` → `ollama pull llama3.1:8b` → calibrate tok/s
  → run nim_b0_anchor → T1 → T2.

---

### 2026-05-08 — (no run_id) — RTX 5080 PC bring-up session

- Backend: n/a (harness + environment checks only)
- Model: n/a
- N per cell: n/a
- Wallclock: ~5 min
- Outcome: Confirmed RTX 5080 Laptop GPU (16 GB VRAM, driver 595.97, CUDA 13.2) is
  visible in WSL2. `uv sync` green. `uv run python -m pytest -q` → **30/30 pass**
  (26 original + 4 new). T1–T6 YAML configs created in `configs/nim_t*.yaml`.
  `scripts/bootstrap_linux.sh` written with Ollama install + model pull + smoke-test
  sequence. Ollama install requires interactive sudo — must run from a real terminal.
- Followups: Run `bash scripts/bootstrap_linux.sh` in a terminal to complete the
  GPU stack setup. Then T1 (nim_t1_step_budget_n85) → T2 pair → stop-and-reassess.

---

### 2026-04-30 — run_3b5e0e753407 — smoke_n3_mock (harness validation)

- Backend: mock
- Model: mock-r1
- N per cell: 3
- Wallclock: 0:00:48
- Outcome: 12/12 games completed end-to-end (sweep → DB → JSONL → plots).
  As expected for the deterministic mock, it loses every game to UCT-50
  (win rate 0% at every B). Pass-1 token counts track B closely
  (0 / 64.0 / 247.7 / 1001.5 mean). Confirms the harness, schema,
  analytics, and plots all work without an LLM serving.
- Followups: run real Ollama sweep (`configs/smoke_n3.yaml`) once
  `bootstrap_mac.sh` has installed openjdk@17 + Ollama + deepseek-r1:7b.
