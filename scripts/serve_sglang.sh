#!/usr/bin/env bash
# serve_sglang.sh — RTX 5080 (Blackwell, sm_120) verified launcher.
#
# VERIFIED COMMAND (2026-05-10, RTX 5080, WSL2, SGLang 0.5.9):
#
#   Llama 3.1 8B Instruct — FP8 quantization (fits 16 GB VRAM):
#     CUDA_HOME=/usr/local/cuda-13.2 uv run python -m sglang.launch_server \
#       --model-path meta-llama/Llama-3.1-8B-Instruct \
#       --port 30000 --host 127.0.0.1 \
#       --quantization fp8
#
# Notes:
#   - CUDA_HOME must be set explicitly; WSL2 doesn't always have it on PATH.
#   - --quantization fp8 is required to fit the 16 GB bfloat16 model in 16 GB VRAM.
#     bfloat16 / float16 load crashes with sigquit (OOM) during shard load.
#   - --host 127.0.0.1 (not 0.0.0.0) — match base_url in experiment configs.
#   - No --context-length needed with fp8; KV cache headroom is sufficient.
#   - No --dtype flag; SGLang infers native dtype and applies fp8 quantization.
#   - Wait for "Server is ready" in the log before starting sweeps (~30-60 s).
#
# DeepSeek-R1-Distill-Qwen-7B (original template, awq_marlin, not yet verified
# on this rig — validate before use):
set -euo pipefail

MODEL="${SGLANG_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
PORT="${SGLANG_PORT:-30000}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.85}"
MAX_RUNNING="${SGLANG_MAX_RUNNING:-16}"

CUDA_HOME=/usr/local/cuda-13.2 python -m sglang.launch_server \
  --model "$MODEL" \
  --quantization awq_marlin \
  --host 0.0.0.0 \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRAC" \
  --max-running-requests "$MAX_RUNNING" \
  --attention-backend flashinfer
