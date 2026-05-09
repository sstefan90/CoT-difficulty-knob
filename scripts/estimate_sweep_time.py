"""Estimate wall-clock time for a sweep before running it.

Reads any SweepConfig YAML and prints a per-cell and total time estimate.
When ``data/results.db`` exists it calibrates per-turn latency from recent
LLM runs at matching budgets; otherwise it falls back to a token-throughput
model with configurable TPS.

Usage::

    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml
    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml --tps 80   # RTX 5080 / SGLang
    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml --db data/results.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cot_knob.experiments.config import load_sweep_config

# ---------------------------------------------------------------------------
# Physical constants & defaults
# ---------------------------------------------------------------------------

# Reversi averages ~60 total plies; LLM plays half.
AVG_TOTAL_TURNS = 60
AVG_LLM_TURNS = 30      # LLM plays every other turn (plays one side)
AVG_UCT_TURNS = 30

# Oracle (UCT-2000) overhead per turn — now runs on EVERY turn (both agents).
# Measured at ~0.25–0.35s on Apple M-series.
ORACLE_S_PER_TURN = 0.30

# UCT opponent overhead per turn at 10 iters vs 2000 iters.
def uct_s_per_turn(iterations: int) -> float:
    # Empirically: UCT-10 ~0.01s, UCT-2000 ~0.3s (same as oracle).
    # Linear approximation is good enough for scheduling.
    return max(0.005, iterations / 7_000)


# ---------------------------------------------------------------------------
# Calibration from results.db
# ---------------------------------------------------------------------------

def _calibrate_from_db(db_path: Path, budget: int) -> float | None:
    """Return mean per-turn LLM latency (seconds) for this budget from history.

    Queries recent LLM turns at the given budget.  Returns None if no data.
    """
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        # latency_ms_total covers pass-1 + pass-2 for LLM turns.
        rows = conn.execute(
            """
            SELECT AVG(t.latency_ms_total)
            FROM turns t
            JOIN trials tr ON t.trial_id = tr.trial_id
            WHERE t.agent_kind = 'llm'
              AND json_extract(tr.condition_json, '$.budget') = ?
              AND t.latency_ms_total > 0
            LIMIT 1
            """,
            (budget,),
        ).fetchone()
        conn.close()
        val = rows[0] if rows else None
        if val:
            return val / 1000.0  # ms → s
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Estimation model
# ---------------------------------------------------------------------------

@dataclass
class CellEstimate:
    budget: int
    n_games: int
    llm_s_per_turn: float      # calibrated or modelled
    llm_source: str            # "calibrated" | "model"
    oracle_s_per_turn: float
    uct_opp_s_per_turn: float

    @property
    def per_game_s(self) -> float:
        # Oracle now runs every turn (both LLM and UCT sides).
        llm_side = AVG_LLM_TURNS * (self.llm_s_per_turn + self.oracle_s_per_turn)
        uct_side = AVG_UCT_TURNS * (self.uct_opp_s_per_turn + self.oracle_s_per_turn)
        return llm_side + uct_side

    @property
    def total_s(self) -> float:
        return self.per_game_s * self.n_games


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate wall-clock time for a sweep config",
    )
    parser.add_argument("config", help="Path to sweep YAML config")
    parser.add_argument(
        "--tps", type=float, default=None,
        help=(
            "LLM tokens-per-second for Pass-1 generation. "
            "Default: 13 (Ollama, Apple M-series) or 80 (SGLang, RTX 5080). "
            "Auto-selected from config backend if not given."
        ),
    )
    parser.add_argument(
        "--db", type=Path,
        default=Path("data/results.db"),
        help="Path to results.db for empirical calibration (default: data/results.db)",
    )
    parser.add_argument(
        "--avg-turns", type=int, default=AVG_TOTAL_TURNS,
        help=f"Expected total turns per game (default: {AVG_TOTAL_TURNS})",
    )
    args = parser.parse_args()

    cfg = load_sweep_config(args.config)
    seeds = cfg.resolved_seeds()
    n_per_cell = len(seeds)
    n_cells = len(cfg.budgets) * n_per_cell

    # Choose default TPS based on backend.
    if args.tps is not None:
        default_tps = args.tps
        tps_source = "cli flag"
    elif cfg.model.backend == "ollama":
        default_tps = 13.0
        tps_source = "Ollama default (Apple M-series ~13 tok/s)"
    elif cfg.model.backend == "sglang":
        default_tps = 80.0
        tps_source = "SGLang default (RTX 5080 INT4 ~80 tok/s)"
    else:
        default_tps = 13.0
        tps_source = "mock/unknown backend, using 13 tok/s"

    uct_opp_s = uct_s_per_turn(cfg.opponent.iterations)
    oracle_s = ORACLE_S_PER_TURN  # same for all cells (oracle_iterations fixed)

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  Sweep ETA estimate:  {args.config}")
    print("=" * 64)
    print(f"  run_name       : {cfg.run_name}")
    print(f"  backend        : {cfg.model.backend}  ({tps_source})")
    print(f"  budgets        : {cfg.budgets}")
    print(f"  n_per_cell     : {n_per_cell}  (seeds: {seeds})")
    print(f"  total cells    : {n_cells}  ({len(cfg.budgets)} budgets × {n_per_cell} seeds)")
    print(f"  opponent       : UCT-{cfg.opponent.iterations}  (~{uct_opp_s:.3f}s/turn)")
    print(f"  oracle         : UCT-{cfg.oracle_iterations}  (~{oracle_s:.2f}s/turn, every turn)")
    print(f"  avg turns/game : {args.avg_turns}  ({args.avg_turns//2} LLM + {args.avg_turns//2} UCT)")
    print()

    # ── Per-budget table ─────────────────────────────────────────────────────
    cells: list[CellEstimate] = []
    print(f"  {'Budget':>7}  {'LLM s/turn':>10}  {'Source':>12}  {'Per game':>9}  {'×N games':>8}  {'Cell total':>10}")
    print("  " + "-" * 62)

    for b in cfg.budgets:
        calibrated = _calibrate_from_db(args.db, b)
        if calibrated is not None:
            llm_s = calibrated
            src = "calibrated"
        else:
            # Model: expected output tokens ≈ min(budget, 0.85*budget+preamble).
            # At most budgets the model hits length, so avg ≈ budget tokens out.
            avg_out_tokens = float(b) if b > 0 else 5.0   # B=0: pass-2 only
            llm_s = avg_out_tokens / default_tps + 0.5    # +0.5s for pass-2 overhead
            src = "model"

        ce = CellEstimate(
            budget=b, n_games=n_per_cell,
            llm_s_per_turn=llm_s, llm_source=src,
            oracle_s_per_turn=oracle_s,
            uct_opp_s_per_turn=uct_opp_s,
        )
        cells.append(ce)

        print(
            f"  {b:>7}  {llm_s:>9.1f}s  {src:>12}  "
            f"{ce.per_game_s:>8.0f}s  {n_per_cell:>8}  {_format_duration(ce.total_s):>10}"
        )

    total_s = sum(c.total_s for c in cells)
    total_games = n_cells

    print()
    print(f"  {'TOTAL':>7}  {'':>10}  {'':>12}  {'':>9}  {total_games:>8}  {_format_duration(total_s):>10}")
    print()

    # ── Breakdown of time sources ─────────────────────────────────────────
    # Highest-budget cell drives total; show the dominant cost.
    if cells:
        slowest = max(cells, key=lambda c: c.llm_s_per_turn)
        llm_frac = (slowest.llm_s_per_turn * AVG_LLM_TURNS) / slowest.per_game_s
        oracle_frac = (oracle_s * AVG_TOTAL_TURNS) / slowest.per_game_s
        uct_frac = (uct_opp_s * AVG_UCT_TURNS) / slowest.per_game_s
        print(f"  At B={slowest.budget}: LLM pass-1 {llm_frac*100:.0f}%  |  oracle {oracle_frac*100:.0f}%  |  UCT opp {uct_frac*100:.0f}%")

    # ── GPU speedup hint ──────────────────────────────────────────────────
    if cfg.model.backend == "ollama":
        gpu_tps = 80.0
        gpu_cells = []
        for ce in cells:
            avg_out = float(ce.budget) if ce.budget > 0 else 5.0
            gpu_llm_s = avg_out / gpu_tps + 0.5
            gpu_ce = CellEstimate(
                budget=ce.budget, n_games=ce.n_games,
                llm_s_per_turn=gpu_llm_s, llm_source="model",
                oracle_s_per_turn=oracle_s,
                uct_opp_s_per_turn=uct_opp_s,
            )
            gpu_cells.append(gpu_ce)
        gpu_total = sum(c.total_s for c in gpu_cells)
        print(f"  GPU estimate (SGLang ~{gpu_tps:.0f} tok/s): {_format_duration(gpu_total)}"
              f"  ({total_s/gpu_total:.1f}× faster than Ollama model)")

    print()

    # ── Calibration tip ──────────────────────────────────────────────────
    n_calibrated = sum(1 for c in cells if c.llm_source == "calibrated")
    if n_calibrated < len(cells):
        not_cal = [c.budget for c in cells if c.llm_source != "calibrated"]
        print(
            f"  Tip: {len(not_cal)} budget(s) {not_cal} have no historical data in {args.db}.\n"
            f"  Run a short smoke test (n_per_cell=1) to get empirical calibration.\n"
        )
    else:
        print(f"  All {len(cells)} budgets calibrated from {args.db}.\n")


if __name__ == "__main__":
    main()
