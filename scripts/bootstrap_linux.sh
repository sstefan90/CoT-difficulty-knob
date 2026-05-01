#!/usr/bin/env bash
# bootstrap_linux.sh — RTX 5090 box setup. Stub for now; expanded after Mac smoke.
#
# Plan: install OpenJDK 17, fetch Ludii.jar, install uv, sync deps with ludii
# extra, install SGLang (or vLLM as a fallback for sm_120 / Blackwell).
set -euo pipefail

echo "Linux bootstrap not implemented yet."
echo "When the smoke sweep is green on Mac, this script will:"
echo "  1. apt install openjdk-17-jdk python3.11"
echo "  2. install uv and run uv sync --extra ludii"
echo "  3. download Ludii.jar"
echo "  4. install SGLang (with sm_120 verification) OR vLLM as fallback"
echo "  5. download DeepSeek-R1-Distill-Qwen-7B INT4 weights"
echo "  6. emit a serve_sglang.sh / serve_vllm.sh launcher with tuned flags."
exit 0
