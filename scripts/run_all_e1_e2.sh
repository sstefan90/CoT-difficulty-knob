#!/usr/bin/env bash
# All Avalon experiments — unified Bayesian servant bot design.
#
# E2: Servant/minimal budget curve (5 points, 150 games) — primary coherence experiment
# E1: Servant/procedural (60 games) + Merlin/minimal + Merlin/procedural (60 games)
#
# All conditions: shuffle_all_roles=True, bot_strategy=bayesian, summarizer=anthropic/sonnet
# Total: 330 games

set -euo pipefail
cd "$(dirname "$0")/.."

# Load API keys (ANTHROPIC_API_KEY needed for Sonnet summarizer)
if [ -f .env ]; then
  set -a; source .env; set +a
fi

LOG="data/sweep_run.log"
mkdir -p data/runs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Starting unified Bayesian sweep (330 games) ==="

CONFIGS=(
    # E2 — Servant/minimal, 5 budget points (150 games) — coherence curve + AvalonBench baseline
    configs/avalon_e2_servant_min_b64_n30_bayesian.yaml
    configs/avalon_e2_servant_min_b128_n30_bayesian.yaml
    configs/avalon_e2_servant_min_b256_n30_bayesian.yaml
    configs/avalon_e2_servant_min_b512_n30_bayesian.yaml
    configs/avalon_e2_servant_min_b1024_n30_bayesian.yaml
    # E1 — Servant/procedural (60 games)
    configs/avalon_e1_servant_proc_n30_shuffled.yaml
    # E1 — Merlin (60 games)
    configs/avalon_e1_merlin_min_n30_shuffled.yaml
    configs/avalon_e1_merlin_proc_n30_shuffled.yaml
)

for cfg in "${CONFIGS[@]}"; do
    log "--- Starting: $cfg ---"
    if python scripts/run_avalon_sweep.py "$cfg" 2>&1 | tee -a "$LOG"; then
        log "--- DONE: $cfg ---"
    else
        log "--- ERROR in $cfg (exit $?) --- continuing ---"
    fi
done

log "=== All sweeps complete ==="
