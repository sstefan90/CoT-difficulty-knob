"""CLI entry point for Avalon budget sweeps.

Usage:
    python scripts/run_avalon_sweep.py configs/avalon_pilot_servant_n10.yaml
    python scripts/run_avalon_sweep.py configs/avalon_e1_servant_n50_minimal.yaml --db data/results.db
"""

import sys
from pathlib import Path

# Make src/ importable when running directly (not installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cot_knob.experiments.avalon_sweep import main_cli

if __name__ == "__main__":
    main_cli()
