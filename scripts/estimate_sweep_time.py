"""Estimate wall-clock time for a sweep before running it.

Reads any SweepConfig YAML and prints a per-cell and total time estimate.
When ``data/results.db`` exists it calibrates per-turn latency from recent
LLM runs at matching budgets; otherwise it falls back to a token-throughput
model with configurable TPS.

Usage::

    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml
    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml --tps 200  # RTX 5080 Ollama
    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml --tps 400  # RTX 5080 batched
    uv run python scripts/estimate_sweep_time.py configs/my_sweep.yaml --db data/results.db

Game-length defaults (auto-detected from config, override with --avg-turns):
    nim   [3,5,7]   →  10 turns/game
    nim   [7,11,13] →  25 turns/game (more objects, more turns)
    reversi         →  60 turns/game
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
# Nim [3,5,7] averages ~10 turns; Nim [7,11,13] averages ~25 turns.
# These are the per-game fallback defaults; overridable with --avg-turns.
_AVG_TURNS_BY_GAME: dict[str, int] = {
    "reversi": 60,
    "nim_small": 10,   # piles like [3,5,7] — total objects ~15
    "nim_large": 25,   # piles like [7,11,13] — total objects ~31
}

# Oracle (UCT-2000) overhead per turn — runs on EVERY turn (both agents).
# Measured at ~0.25–0.35s on Apple M-series.  Zero when oracle_iterations=0.
ORACLE_S_PER_TURN = 0.30

# UCT opponent overhead per turn at 10 iters vs 2000 iters.
def uct_s_per_turn(iterations: int) -> float:
    # Empirically: UCT-10 ~0.01s, UCT-2000 ~0.3s (same as oracle).
    # Linear approximation is good enough for scheduling.
    return max(0.005, iterations / 7_000)


def _default_avg_turns(cfg) -> int:
    """Infer a sensible default avg-turns from the sweep config."""
    if cfg.game == "nim":
        total_objects = sum(cfg.nim_piles)
        return _AVG_TURNS_BY_GAME["nim_large"] if total_objects > 20 else _AVG_TURNS_BY_GAME["nim_small"]
    return _AVG_TURNS_BY_GAME["reversi"]


def _oracle_s_per_turn(cfg) -> float:
    """Oracle cost per turn. Zero when oracle_iterations=0 (most Nim configs)."""
    return ORACLE_S_PER_TURN if cfg.oracle_iterations > 0 else 0.0


def _opp_s_per_turn(cfg) -> float:
    """UCT opponent cost per turn.  Near-zero for random / nim_optimal opponents."""
    if cfg.opponent.kind == "uct":
        return uct_s_per_turn(cfg.opponent.iterations)
    return 0.005   # random / nim_optimal: essentially free


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
    avg_llm_turns: int
    avg_opp_turns: int

    @property
    def per_game_s(self) -> float:
        # Oracle runs every turn (both LLM and opp sides) when enabled.
        llm_side = self.avg_llm_turns * (self.llm_s_per_turn + self.oracle_s_per_turn)
        opp_side = self.avg_opp_turns * (self.uct_opp_s_per_turn + self.oracle_s_per_turn)
        return llm_side + opp_side

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
            "Defaults: Ollama/Mac ~13, Ollama/RTX-5080 ~150-300, SGLang/RTX-5080 ~400. "
            "Auto-selected from config backend if not given."
        ),
    )
    parser.add_argument(
        "--db", type=Path,
        default=Path("data/results.db"),
        help="Path to results.db for empirical calibration (default: data/results.db)",
    )
    parser.add_argument(
        "--avg-turns", type=int, default=None,
        help=(
            "Expected total turns per game. "
            "Auto-detected from game type if omitted: "
            "nim[3,5,7]→10, nim[7,11,13]→25, reversi→60."
        ),
    )
    args = parser.parse_args()

    cfg = load_sweep_config(args.config)
    seeds = cfg.resolved_seeds()
    n_per_cell = len(seeds)
    n_cells = len(cfg.budgets) * n_per_cell

    # Choose default TPS based on backend.
    if args.tps is not None:
        default_tps = args.tps
        tps_source = f"cli flag ({args.tps:.0f} tok/s)"
    elif cfg.model.backend == "ollama":
        default_tps = 13.0
        tps_source = "Ollama default (Mac ~13; pass --tps 200 for RTX 5080 estimate)"
    elif cfg.model.backend == "sglang":
        default_tps = 400.0
        tps_source = "SGLang default (RTX 5080 ~400 tok/s)"
    else:
        default_tps = 13.0
        tps_source = "mock/unknown backend, using 13 tok/s"

    # Correct overhead per turn for this config's actual opponent + oracle settings.
    uct_opp_s = _opp_s_per_turn(cfg)
    oracle_s = _oracle_s_per_turn(cfg)

    # Auto-detect avg turns from game type, or use CLI override.
    avg_total_turns = args.avg_turns if args.avg_turns is not None else _default_avg_turns(cfg)
    avg_llm_turns = avg_total_turns // 2
    avg_opp_turns = avg_total_turns - avg_llm_turns

    opp_label = (
        f"UCT-{cfg.opponent.iterations}" if cfg.opponent.kind == "uct"
        else cfg.opponent.kind
    )
    oracle_label = (
        f"UCT-{cfg.oracle_iterations}  (~{oracle_s:.2f}s/turn, every turn)"
        if cfg.oracle_iterations > 0
        else "disabled (oracle_iterations=0)"
    )

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  Sweep ETA estimate:  {args.config}")
    print("=" * 64)
    print(f"  run_name       : {cfg.run_name}")
    print(f"  game           : {cfg.game}" + (f"  piles={cfg.nim_piles}" if cfg.game == "nim" else ""))
    print(f"  backend        : {cfg.model.backend}  ({tps_source})")
    print(f"  budgets        : {cfg.budgets}")
    print(f"  n_per_cell     : {n_per_cell}")
    print(f"  total games    : {n_cells}  ({len(cfg.budgets)} budgets × {n_per_cell} games/cell)")
    print(f"  opponent       : {opp_label}  (~{uct_opp_s:.3f}s/turn)")
    print(f"  oracle         : {oracle_label}")
    print(f"  avg turns/game : {avg_total_turns}  ({avg_llm_turns} LLM + {avg_opp_turns} opp)")
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
            avg_llm_turns=avg_llm_turns,
            avg_opp_turns=avg_opp_turns,
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
    if cells:
        slowest = max(cells, key=lambda c: c.llm_s_per_turn)
        pg = slowest.per_game_s
        if pg > 0:
            llm_frac = (slowest.llm_s_per_turn * avg_llm_turns) / pg
            oracle_frac = (oracle_s * avg_total_turns) / pg
            opp_frac = (uct_opp_s * avg_opp_turns) / pg
            print(f"  At B={slowest.budget}: LLM {llm_frac*100:.0f}%  |  oracle {oracle_frac*100:.0f}%  |  opp {opp_frac*100:.0f}%")

    # ── GPU speedup hints ─────────────────────────────────────────────────
    if cfg.model.backend == "ollama" and args.tps is None:
        print()
        for label, gpu_tps in [("Ollama/RTX-5080 est. (~200 tok/s)", 200.0),
                                ("SGLang/RTX-5080 est. (~400 tok/s)", 400.0)]:
            gpu_cells = []
            for ce in cells:
                avg_out = float(ce.budget) if ce.budget > 0 else 5.0
                gpu_llm_s = avg_out / gpu_tps + 0.5
                gpu_ce = CellEstimate(
                    budget=ce.budget, n_games=ce.n_games,
                    llm_s_per_turn=gpu_llm_s, llm_source="model",
                    oracle_s_per_turn=oracle_s,
                    uct_opp_s_per_turn=uct_opp_s,
                    avg_llm_turns=avg_llm_turns,
                    avg_opp_turns=avg_opp_turns,
                )
                gpu_cells.append(gpu_ce)
            gpu_total = sum(c.total_s for c in gpu_cells)
            speedup = f"  ({total_s/gpu_total:.1f}× faster than Mac Ollama)" if total_s > gpu_total else ""
            print(f"  {label}: {_format_duration(gpu_total)}{speedup}")

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
