#!/usr/bin/env bash
# bootstrap_mac.sh — one-shot Mac dev setup for the CoT-knob harness.
#
# Idempotent: safe to re-run. Installs (only what's missing):
#   - Homebrew openjdk@17  (required for Ludii via JPype)
#   - Homebrew ant         (jpype1 1.7+ builds from source; CMake invokes `ant`)
#   - Ollama               (Mac LLM backend)
#   - Ollama model: deepseek-r1:7b (DeepSeek-R1-Distill-Qwen-7B, ~4.7 GB)
#   - Ludii.jar v1.3.14    (game engine + UCT baseline)
#   - Python deps          (uv sync --extra dev --extra ludii)
#
# Usage:  bash scripts/bootstrap_mac.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

c_blue() { printf "\033[1;34m%s\033[0m\n" "$*"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
c_yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }

# 1. Homebrew + openjdk@17
c_blue "[1/5] Checking Homebrew + openjdk@17..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh first."
  exit 1
fi
if brew list --formula openjdk@17 >/dev/null 2>&1; then
  c_green "  openjdk@17 already installed."
else
  c_yellow "  Installing openjdk@17 (this can take a few minutes)..."
  brew install openjdk@17
fi
JAVA_HOME_CANDIDATE="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
if [[ ! -d "$JAVA_HOME_CANDIDATE" ]]; then
  echo "Expected JAVA_HOME at $JAVA_HOME_CANDIDATE not found. Check brew install output."
  exit 1
fi
c_green "  JAVA_HOME = $JAVA_HOME_CANDIDATE"

if brew list --formula ant >/dev/null 2>&1; then
  c_green "  ant already installed."
else
  c_yellow "  Installing ant (required to build jpype1 wheel)..."
  brew install ant
fi

# 2. Ollama
c_blue "[2/5] Checking Ollama..."
if command -v ollama >/dev/null 2>&1; then
  c_green "  Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
  c_yellow "  Installing Ollama via Homebrew cask..."
  brew install --cask ollama
fi

# 3. Pull DeepSeek-R1-Distill-Qwen-7B
c_blue "[3/5] Pulling deepseek-r1:7b (DeepSeek-R1-Distill-Qwen-7B, ~4.7 GB)..."
if ! pgrep -x ollama >/dev/null 2>&1; then
  c_yellow "  Starting ollama serve in the background..."
  (ollama serve >/tmp/ollama.log 2>&1 &) || true
  sleep 3
fi
if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^deepseek-r1:7b$"; then
  c_green "  deepseek-r1:7b already pulled."
else
  ollama pull deepseek-r1:7b
fi

# 4. Ludii.jar
c_blue "[4/5] Downloading Ludii.jar v1.3.14..."
LUDII_JAR="$REPO_ROOT/third_party/Ludii.jar"
mkdir -p "$REPO_ROOT/third_party"
if [[ -f "$LUDII_JAR" ]]; then
  c_green "  Ludii.jar already present."
else
  # Ludii is hosted at https://ludii.games/downloads/Ludii-1.3.14.jar
  # Note: re-check the version on https://ludii.games/download.php if this URL 404s.
  LUDII_URL="https://ludii.games/downloads/Ludii-1.3.14.jar"
  c_yellow "  Fetching $LUDII_URL ..."
  curl -L --fail "$LUDII_URL" -o "$LUDII_JAR"
fi

# 5. Python deps
c_blue "[5/5] uv sync (Python 3.11 + deps + jpype1)..."
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv/python"
export UV_CACHE_DIR="$REPO_ROOT/.uv/cache"
export JAVA_HOME="$JAVA_HOME_CANDIDATE"
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"
uv sync --extra dev --extra ludii

c_green "All set. Next steps:"
cat <<EOF

  # 1. Activate the venv (or use 'uv run <cmd>' for one-offs):
  source .venv/bin/activate

  # 2. Set env (or copy .env.example -> .env):
  export JAVA_HOME="$JAVA_HOME_CANDIDATE"
  export LUDII_JAR="$LUDII_JAR"
  export OLLAMA_URL=http://localhost:11434
  export OLLAMA_MODEL=deepseek-r1:7b

  # 3. Probe the Ludii bridge:
  uv run python -m cot_knob.games.ludii_bridge --probe

  # 4. Run the smoke sweep:
  uv run python scripts/run_budget_sweep.py configs/smoke_n3.yaml
EOF
