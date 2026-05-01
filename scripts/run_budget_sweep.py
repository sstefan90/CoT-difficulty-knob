"""Thin wrapper that calls cot_knob.experiments.sweep.main_cli().

Run as::

    uv run python scripts/run_budget_sweep.py configs/smoke_n3.yaml
"""

from __future__ import annotations

from cot_knob.experiments.sweep import main_cli

if __name__ == "__main__":
    main_cli()
