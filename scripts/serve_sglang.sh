#!/usr/bin/env bash
# serve_sglang.sh — RTX 5090 launcher template. Validate before running on prod.
#
# WARNING: As of early 2026, SGLang has open issues with INT4/marlin kernels
# on sm_120 (Blackwell). Verify with a short sanity run before committing
# multi-day sweeps. If SGLang misbehaves, swap to serve_vllm.sh (TBD).
set -euo pipefail

MODEL="${SGLANG_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
PORT="${SGLANG_PORT:-30000}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.85}"
MAX_RUNNING="${SGLANG_MAX_RUNNING:-16}"

python -m sglang.launch_server \
  --model "$MODEL" \
  --quantization awq_marlin \
  --host 0.0.0.0 \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRAC" \
  --max-running-requests "$MAX_RUNNING" \
  --attention-backend flashinfer
