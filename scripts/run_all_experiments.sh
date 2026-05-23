#!/usr/bin/env bash
# Run all Avalon experiment configs in priority order.
# Output is tee'd to data/sweep.log so you can tail -f it in another terminal.
#
# Usage: bash scripts/run_all_experiments.sh [--db data/results.db] [--runs data/runs]
#
# P2-A  (N=50 Servant, must complete before P2-B analysis):
#   avalon_e1_servant_n50_minimal     -> cells L-S-min-low  + L-S-min-high
#   avalon_e1_servant_n50_procedural  -> cells L-S-proc-low + L-S-proc-high
#
# P2-B  (N=30, Merlin + discussion):
#   avalon_e1_merlin_n30              -> cell  L-M-min-high
#   avalon_e1_merlin_n30_procedural   -> cell  L-M-proc-high
#   avalon_e1_servant_n30_minimal_disc     -> cell L-S-min-disc
#   avalon_e1_servant_n30_minimal_disc_low -> cell L-S-min-disc-low
#   avalon_e1_merlin_n30_minimal_disc      -> cell L-M-min-disc

set -euo pipefail

DB="${1:---db}"
DB_PATH="${2:-data/results.db}"
RUNS_PATH="data/runs"
LOG="data/sweep.log"

# Override from env if caller passes --db / --runs flags the normal way
for arg in "$@"; do
  case "$arg" in
    --db=*) DB_PATH="${arg#--db=}" ;;
    --runs=*) RUNS_PATH="${arg#--runs=}" ;;
  esac
done

CONFIGS=(
    configs/avalon_e1_servant_n50_minimal.yaml
    configs/avalon_e1_servant_n50_procedural.yaml
    configs/avalon_e1_merlin_n30.yaml
    configs/avalon_e1_merlin_n30_procedural.yaml
    configs/avalon_e1_servant_n30_minimal_disc.yaml
    configs/avalon_e1_servant_n30_minimal_disc_low.yaml
    configs/avalon_e1_merlin_n30_minimal_disc.yaml
)

mkdir -p data
: > "$LOG"   # truncate / create log

total=${#CONFIGS[@]}
idx=0

for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo "" | tee -a "$LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "[$idx/$total] $(date '+%H:%M:%S')  START  $cfg" | tee -a "$LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"

    start_ts=$(date +%s)

    if uv run python scripts/run_avalon_sweep.py "$cfg" \
            --db "$DB_PATH" --runs "$RUNS_PATH" 2>&1 | tee -a "$LOG"; then
        status="OK"
    else
        status="FAILED (exit $?)"
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    echo "" | tee -a "$LOG"
    echo "[$idx/$total] $(date '+%H:%M:%S')  $status  $cfg  (${elapsed}s)" | tee -a "$LOG"

    if [[ "$status" != "OK" ]]; then
        echo "ERROR: sweep failed for $cfg — aborting." | tee -a "$LOG"
        exit 1
    fi
done

echo "" | tee -a "$LOG"
echo "All $total configs completed successfully.  Log: $LOG" | tee -a "$LOG"
