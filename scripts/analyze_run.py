"""Produce a quick analysis of a sweep run.

Usage::

    uv run python scripts/analyze_run.py <run_id> [--db data/results.db]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from cot_knob.analysis.plots import (
    plot_latency_and_agreement,
    plot_token_count_sanity,
    plot_win_rate_vs_budget,
)
from cot_knob.tracking.analytics import (
    latency_stats_by_budget,
    open_ro,
    reasoning_token_stats_by_budget,
    resolve_run_id,
    uct_top3_agreement_by_budget,
    win_rate_by_budget,
)

console = Console()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--db", default="data/results.db")
    p.add_argument("--out", default=None, help="Output dir; default data/runs/<run_id>/figures")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else Path("data/runs") / args.run_id / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = open_ro(args.db)
    try:
        resolved = resolve_run_id(conn, args.run_id)
        if resolved != args.run_id:
            console.print(f"[dim]Resolved run_id: {args.run_id!r} → {resolved!r}[/dim]")
        args.run_id = resolved
        wr = win_rate_by_budget(conn, args.run_id)
        ts = reasoning_token_stats_by_budget(conn, args.run_id)
        lat = latency_stats_by_budget(conn, args.run_id)
        agr = uct_top3_agreement_by_budget(conn, args.run_id)
    finally:
        conn.close()

    if not wr:
        console.print(f"[red]No trials found for run_id={args.run_id}[/red]")
        return

    table = Table(title=f"Sweep summary — run {args.run_id}")
    table.add_column("B", justify="right")
    table.add_column("N", justify="right")
    table.add_column("wins", justify="right")
    table.add_column("win_rate", justify="right")
    table.add_column("Wilson 95% CI")
    table.add_column("mean tok_out")
    table.add_column("mean lat (ms)")
    table.add_column("UCT-top3 hit")
    for b, row in wr.items():
        ci = f"[{row['wilson_low']:.2f}, {row['wilson_high']:.2f}]"
        table.add_row(
            str(b), str(row["n"]), str(row["wins"]),
            f"{row['win_rate']:.2f}", ci,
            f"{ts.get(b, {}).get('mean_tokens_out', 0):.1f}",
            f"{lat.get(b, {}).get('mean_ms', 0):.0f}",
            f"{agr.get(b, {}).get('agreement', 0):.2f}",
        )
    console.print(table)

    # Save raw stats as JSON for easy diffing across runs.
    raw = {"win_rate": wr, "tokens": ts, "latency": lat, "agreement": agr}
    (out_dir / "summary.json").write_text(json.dumps(raw, indent=2, sort_keys=True))

    p1 = plot_win_rate_vs_budget(args.db, args.run_id, out_dir / "win_rate_vs_B.png")
    p2 = plot_token_count_sanity(args.db, args.run_id, out_dir / "tokens_vs_B.png")
    p3 = plot_latency_and_agreement(args.db, args.run_id, out_dir / "latency_and_agreement.png")
    console.print(f"[green]Wrote:[/green] {p1}\n        {p2}\n        {p3}")


if __name__ == "__main__":
    main()
