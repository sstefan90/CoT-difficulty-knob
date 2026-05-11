#!/usr/bin/env bash
# bootstrap_linux.sh — RTX 5080 / WSL2 setup for CoT-difficulty-knob.
#
# Installs: Ollama (GPU-backed llama.cpp), pulls llama3.1:8b + deepseek-r1:7b,
# ensures uv venv is synced, and runs the full smoke-test sequence from
# docs/pc_dev_setup.md §5.
#
# Usage:
#   bash scripts/bootstrap_linux.sh
#
# Prerequisites (must already be present):
#   - NVIDIA driver ≥ 595 (CUDA 13.x on Blackwell; check: nvidia-smi)
#   - uv  (https://docs.astral.sh/uv/getting-started/installation/)
#   - git
#
# This script is idempotent: re-running it is safe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;32m[bootstrap]\033[0m $*"; }
warn() { echo -e "\033[1;33m[bootstrap WARN]\033[0m $*"; }
die()  { echo -e "\033[1;31m[bootstrap ERROR]\033[0m $*" >&2; exit 1; }

# ── 0. Preflight ─────────────────────────────────────────────────────────────
log "Checking NVIDIA driver..."
if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found. Install NVIDIA drivers before running this script."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
log "GPU OK."

if ! command -v uv &>/dev/null; then
  # Try common install location from the uv installer
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    die "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
fi
log "uv $(uv --version) found."

# ── 1. Python deps ────────────────────────────────────────────────────────────
log "Syncing Python dependencies (dev extra)..."
uv sync --extra dev
log "uv sync complete."

# ── 2. Ollama ────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  log "Ollama installed."
else
  log "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
fi

# Start the Ollama daemon if it isn't already running.
# Under WSL2, systemd may not be active; start directly if needed.
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  log "Starting ollama serve in background..."
  # ollama serve writes logs to stderr; redirect and disown
  ollama serve > /tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!
  log "Ollama daemon PID $OLLAMA_PID — waiting for it to come up..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      log "Ollama is ready."
      break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
      die "Ollama did not come up in 30 s. Check /tmp/ollama.log for errors."
    fi
  done
else
  log "Ollama daemon is already running."
fi

# ── 3. Model pulls ───────────────────────────────────────────────────────────
pull_if_missing() {
  local model="$1"
  if ollama list 2>/dev/null | grep -q "^${model}"; then
    log "Model '$model' already present — skipping pull."
  else
    log "Pulling model '$model' (this may take a few minutes)..."
    ollama pull "$model"
    log "Pull complete: $model"
  fi
}

pull_if_missing "llama3.1:8b"
pull_if_missing "deepseek-r1:7b"

# ── 4. Throughput calibration ─────────────────────────────────────────────────
log "Quick throughput calibration (llama3.1:8b, ~200 tokens)..."
ollama run llama3.1:8b \
  "Count from 1 to 200, one number per line." \
  --verbose 2>&1 | tail -5 || warn "Throughput calibration run failed — not critical."

# ── 5. Pytest smoke test ──────────────────────────────────────────────────────
log "Running pytest (mock backend, no GPU required)..."
uv run python -m pytest -q
log "All tests passed."

# ── 6. Mock sweep smoke test ──────────────────────────────────────────────────
log "Running mock sweep (configs/smoke_n3_mock.yaml)..."
uv run python scripts/run_budget_sweep.py configs/smoke_n3_mock.yaml
log "Mock sweep complete."

# ── 7. B=0 anchor (cheapest GPU-touching test) ───────────────────────────────
log "Running B=0 anchor sweep (configs/nim_b0_anchor.yaml)..."
log "Expected: ~50% win rate, ~1 min wallclock."
uv run python scripts/run_budget_sweep.py configs/nim_b0_anchor.yaml
log "B=0 anchor complete."

# ── done ─────────────────────────────────────────────────────────────────────
log ""
log "Bootstrap complete. GPU stack is ready."
log ""
log "Next steps:"
log "  1. Check throughput output above; update nim_test_backlog.md §6."
log "  2. Run T1 (HIGH priority, 255 games):"
log "       uv run python scripts/run_budget_sweep.py configs/nim_t1_step_budget_n85.yaml"
log "  3. Run T2 in parallel (or back-to-back, 170 games each):"
log "       uv run python scripts/run_budget_sweep.py configs/nim_t2_step_n85.yaml"
log "       uv run python scripts/run_budget_sweep.py configs/nim_t2_free_n85.yaml"
log "  4. Analyze results:"
log "       uv run python scripts/analyze_run.py <run_id>"
log ""
log "All T1–T6 configs live in configs/nim_t*.yaml"
log "See docs/nim_test_backlog.md for interpretation guidance."
