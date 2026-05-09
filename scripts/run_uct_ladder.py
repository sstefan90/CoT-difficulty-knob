"""UCT strength ladder benchmark.

Establishes an absolute playing-strength scale by running:

  UCT-{10, 50, 100, 500, 2000}  vs  Random   (N games each side)
  UCT-10 vs UCT-50, UCT-50 vs UCT-100, ..., UCT-500 vs UCT-2000  (cross-matches)

Both sides are played for each match (agent A as Black + White) to cancel out
Reversi's first-mover advantage.

Games run in parallel using ProcessPoolExecutor (one game per CPU core).
Typical wall-clock at N=50 on an 8-core Mac: ~15 min.
On a 24-core server: ~5 min.

Usage::

    uv run python scripts/run_uct_ladder.py                    # N=50, all cores
    uv run python scripts/run_uct_ladder.py --n 20 --jobs 4   # quick calibration
    uv run python scripts/run_uct_ladder.py --out data/uct_ladder.json
    uv run python scripts/run_uct_ladder.py --n 5             # smoke test (~2 min)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Pure-Python game loop (no async — needed for multiprocessing pickling).
# Agents are reconstructed per-process from (kind, iterations) spec tuples.
# ---------------------------------------------------------------------------

def _make_agent(spec: tuple[str, int | None]):
    """Create an agent from a (kind, iterations_or_None) spec."""
    kind, iters = spec
    if kind == "random":
        from cot_knob.agents.random_agent import RandomAgent
        return RandomAgent()
    if kind == "uct":
        from cot_knob.agents.uct_agent import UCTAgent
        return UCTAgent(iterations=iters)  # type: ignore[arg-type]
    raise ValueError(f"Unknown agent kind: {kind!r}")


def _play_game_sync(
    black_spec: tuple[str, int | None],
    white_spec: tuple[str, int | None],
    seed: int,
    max_turns: int = 200,
) -> str:
    """Play one game synchronously; returns 'black', 'white', or 'draw'.

    Agents are re-instantiated inside each worker process from spec tuples so
    the function is picklable without any shared state.
    """
    import asyncio
    from cot_knob.games.reversi import initial_state

    black = _make_agent(black_spec)
    white = _make_agent(white_spec)
    state = initial_state()
    pass_streak = 0

    async def _run() -> str:
        nonlocal pass_streak, state
        for turn in range(max_turns):
            agent = black if state.current_player == 1 else white
            tel = await agent.choose(state, seed=seed * 1000 + turn)
            if tel.chosen_move < 0:
                pass_streak += 1
                state = state.apply_move(-1)
            else:
                pass_streak = 0
                state = state.apply_move(tel.chosen_move)
            if state.is_terminal() or pass_streak >= 2:
                break
        w = state.winner()
        if w == 1:
            return "black"
        if w == -1:
            return "white"
        return "draw"

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    label_a: str
    label_b: str
    n_per_side: int        # total = 2 × n_per_side
    wins_a: int
    wins_b: int
    draws: int
    wallclock_s: float

    @property
    def win_rate_a(self) -> float:
        total = self.wins_a + self.wins_b + self.draws
        return self.wins_a / total if total else 0.0

    @property
    def win_rate_b(self) -> float:
        total = self.wins_a + self.wins_b + self.draws
        return self.wins_b / total if total else 0.0


def _run_match(
    label_a: str,
    spec_a: tuple[str, int | None],
    label_b: str,
    spec_b: tuple[str, int | None],
    *,
    n: int,
    seed_base: int,
    jobs: int,
) -> MatchResult:
    """Play N games as A=Black + N games as A=White in parallel."""
    tasks: list[tuple[tuple, tuple, int, str]] = []
    for i in range(n):
        tasks.append((spec_a, spec_b, seed_base + i, "a_black"))        # A=Black
        tasks.append((spec_b, spec_a, seed_base + n + i, "a_white"))    # A=White

    wins_a = wins_b = draws = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        futures = {
            pool.submit(_play_game_sync, black, white, seed): role
            for (black, white, seed, role) in tasks
        }
        for fut in as_completed(futures):
            role = futures[fut]
            result = fut.result()
            if role == "a_black":
                if result == "black":
                    wins_a += 1
                elif result == "white":
                    wins_b += 1
                else:
                    draws += 1
            else:  # a_white: A is White, B is Black
                if result == "white":
                    wins_a += 1
                elif result == "black":
                    wins_b += 1
                else:
                    draws += 1

    elapsed = time.perf_counter() - t0
    total = 2 * n
    pct_a = wins_a / total * 100
    pct_b = wins_b / total * 100
    print(
        f"  {label_a:10s} vs {label_b:10s}  |"
        f"  {label_a}: {wins_a}/{total} ({pct_a:.1f}%)"
        f"  {label_b}: {wins_b}/{total} ({pct_b:.1f}%)"
        f"  draws: {draws}"
        f"  ({elapsed:.1f}s  {jobs}j)"
    )
    return MatchResult(
        label_a=label_a, label_b=label_b, n_per_side=n,
        wins_a=wins_a, wins_b=wins_b, draws=draws,
        wallclock_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

UCT_LEVELS = [10, 50, 100, 500, 2000]
_SEED_STRIDE = 100_000


def main(n: int, jobs: int, out: Path | None) -> None:
    results: list[MatchResult] = []
    seed = 0

    print(f"\n{'='*72}")
    print(f"UCT Strength Ladder   N={n}/side ({2*n} total per match)   jobs={jobs}")
    print(f"{'='*72}")

    # ── Section 1: UCT-X vs Random ──────────────────────────────────────────
    print("\n[1] UCT-N vs Random  (absolute strength scale)\n")
    for iters in UCT_LEVELS:
        label = f"UCT-{iters}"
        mr = _run_match(
            label, ("uct", iters),
            "Random", ("random", None),
            n=n, seed_base=seed, jobs=jobs,
        )
        results.append(mr)
        seed += _SEED_STRIDE

    # ── Section 2: Adjacent UCT cross-matches ────────────────────────────────
    print("\n[2] UCT-X vs UCT-Y  (monotonicity check)\n")
    for lo, hi in zip(UCT_LEVELS, UCT_LEVELS[1:]):
        mr = _run_match(
            f"UCT-{lo}", ("uct", lo),
            f"UCT-{hi}", ("uct", hi),
            n=n, seed_base=seed, jobs=jobs,
        )
        results.append(mr)
        seed += _SEED_STRIDE

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Match':<26}  {'A win%':>7}  {'B win%':>7}  {'Draws':>5}  {'Secs':>7}")
    print("  " + "-" * 60)
    for mr in results:
        name = f"{mr.label_a} vs {mr.label_b}"
        print(
            f"  {name:<26}  {mr.win_rate_a*100:6.1f}%  {mr.win_rate_b*100:6.1f}%"
            f"  {mr.draws:5d}  {mr.wallclock_s:7.1f}s"
        )

    # ── Strength scale (UCT vs Random bar) ───────────────────────────────────
    vs_random = [r for r in results if r.label_b == "Random"]
    if vs_random:
        print(f"\n  Strength scale  (win rate vs Random)\n  {'─'*42}")
        for r in vs_random:
            bar = "█" * int(r.win_rate_a * 30) + "░" * (30 - int(r.win_rate_a * 30))
            print(f"  {r.label_a:10s}  {bar}  {r.win_rate_a*100:5.1f}%")

    total_elapsed = sum(r.wallclock_s for r in results)
    total_games = sum(2 * r.n_per_side for r in results)
    print(f"\n  Total: {total_games} games in {total_elapsed:.1f}s "
          f"({total_elapsed/total_games:.2f}s/game wall-clock)\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "label_a": r.label_a, "label_b": r.label_b,
                "n_per_side": r.n_per_side,
                "total_games": 2 * r.n_per_side,
                "wins_a": r.wins_a, "wins_b": r.wins_b, "draws": r.draws,
                "win_rate_a": round(r.win_rate_a, 4),
                "win_rate_b": round(r.win_rate_b, 4),
                "wallclock_s": round(r.wallclock_s, 2),
            }
            for r in results
        ]
        out.write_text(json.dumps(payload, indent=2))
        print(f"  Results saved → {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCT strength ladder benchmark")
    parser.add_argument(
        "--n", type=int, default=50,
        help="Games per side per match (total = 2×N per match). Default: 50",
    )
    parser.add_argument(
        "--jobs", "-j", type=int,
        default=max(1, os.cpu_count() or 4),
        help="Parallel worker processes. Default: all CPU cores",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write results JSON to this path (e.g. data/uct_ladder.json)",
    )
    args = parser.parse_args()
    main(n=args.n, jobs=args.jobs, out=args.out)
