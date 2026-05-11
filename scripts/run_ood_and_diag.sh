#!/usr/bin/env bash
# run_ood_and_diag.sh
# Run OOD probe (Step 1) then diagnostic sweeps (Step 2).
# Supports --backend ollama (default) or --backend sglang.
#
# Usage:
#   bash scripts/run_ood_and_diag.sh                   # Ollama
#   bash scripts/run_ood_and_diag.sh --backend sglang  # SGLang + constrained decoding
#
# Logs land in data/probes/ and data/runs/ as usual.

set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-ollama}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
MODEL="${MODEL:-default}"
N_POSITIONS="${N_POSITIONS:-8}"

# parse --backend arg
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "========================================================"
echo "Backend: $BACKEND"
echo "========================================================"

if [[ "$BACKEND" == "sglang" ]]; then
  echo "Checking SGLang server at $SGLANG_URL ..."
  curl -sf "$SGLANG_URL/health" > /dev/null 2>&1 || \
  curl -sf "$SGLANG_URL/v1/models" > /dev/null 2>&1 || {
    echo "ERROR: SGLang not reachable at $SGLANG_URL — start server first."
    exit 1
  }
  DIAG_SUFFIX="_sglang"
else
  echo "Checking Ollama server at $OLLAMA_URL ..."
  curl -sf "$OLLAMA_URL/api/tags" > /dev/null || {
    echo "ERROR: Ollama not reachable at $OLLAMA_URL — run: sudo systemctl start ollama"
    exit 1
  }
  DIAG_SUFFIX=""
fi
echo ""

# ── Step 1: OOD Probe (strategy only) ────────────────────────────────────────
if [[ "$BACKEND" == "sglang" ]]; then
  echo "========================================================"
  echo "Step 1a: OOD Probe 2 (strategy) — SGLang UNCONSTRAINED"
  echo "========================================================"
  uv run python -u scripts/probe_nim_ood.py \
    --backend sglang --sglang-url "$SGLANG_URL" --model "$MODEL" \
    --no-legality --n-positions "$N_POSITIONS" \
    --out data/probes/nim_ood.jsonl

  echo ""
  echo "========================================================"
  echo "Step 1b: OOD Probe 2 (strategy) — SGLang CONSTRAINED"
  echo "========================================================"
  uv run python -u scripts/probe_nim_ood.py \
    --backend sglang --sglang-url "$SGLANG_URL" --model "$MODEL" \
    --constrained --n-positions "$N_POSITIONS" \
    --out data/probes/nim_ood_constrained.jsonl
else
  echo "========================================================"
  echo "Step 1: OOD Probe 2 (strategy) — Ollama baseline"
  echo "========================================================"
  uv run python -u scripts/probe_nim_ood.py \
    --backend ollama --ollama-url "$OLLAMA_URL" \
    --no-legality --n-positions "$N_POSITIONS" \
    --out data/probes/nim_ood.jsonl
fi

echo ""

# ── Step 2: Diagnostic sweeps (N=30, memory-fixed) ───────────────────────────
echo "========================================================"
echo "Step 2: Diagnostic sweeps (nim_diag_a/b/c + nim_abl_d) N=30"
echo "========================================================"
for base in nim_diag_a_nimsum nim_diag_b_scaffold nim_diag_c_fewshot nim_abl_d_forced_loss; do
  cfg="configs/${base}${DIAG_SUFFIX}.yaml"
  echo ""
  echo "── Running $cfg ──"
  MPLCONFIGDIR=/tmp/mpl uv run python -u scripts/run_budget_sweep.py "$cfg"
done

echo ""
echo "========================================================"
echo "All done. Results:"
echo "  OOD probe:          data/probes/"
echo "  Diagnostic sweeps:  data/runs/ + data/results.db"
echo "========================================================"
