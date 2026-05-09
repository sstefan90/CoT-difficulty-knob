"""Plotting helpers for sweep results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

from cot_knob.tracking.analytics import (  # noqa: E402
    latency_stats_by_budget,
    open_ro,
    reasoning_token_stats_by_budget,
    uct_top3_agreement_by_budget,
    win_rate_by_budget,
)


def plot_win_rate_vs_budget(db_path: str, run_id: str, out_path: str | Path) -> Path:
    conn = open_ro(db_path)
    try:
        wr = win_rate_by_budget(conn, run_id)
    finally:
        conn.close()
    if not wr:
        raise RuntimeError(f"No trials found for run_id={run_id}")

    budgets = list(wr.keys())
    rates = [wr[b]["win_rate"] for b in budgets]
    los = [wr[b]["wilson_low"] for b in budgets]
    his = [wr[b]["wilson_high"] for b in budgets]
    yerr_low = [r - lo for r, lo in zip(rates, los)]
    yerr_hi = [hi - r for hi, r in zip(his, rates)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        budgets, rates, yerr=[yerr_low, yerr_hi],
        fmt="o-", capsize=4, lw=2, label="LLM win rate vs UCT",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% target")
    ax.set_xscale("symlog")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("CoT budget B (max output tokens, Pass 1)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Win rate vs CoT budget — run {run_id}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_token_count_sanity(db_path: str, run_id: str, out_path: str | Path) -> Path:
    conn = open_ro(db_path)
    try:
        ts = reasoning_token_stats_by_budget(conn, run_id)
    finally:
        conn.close()
    budgets = list(ts.keys())
    means = [ts[b]["mean_tokens_out"] for b in budgets]
    p95 = [ts[b]["p95_tokens_out"] for b in budgets]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(budgets, means, "o-", label="mean output tokens (Pass 1)")
    ax.plot(budgets, p95, "x--", label="p95 output tokens")
    ax.plot(budgets, budgets, ":", label="y=x (budget cap)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("CoT budget B")
    ax.set_ylabel("Output tokens (Pass 1)")
    ax.set_title(f"Pass-1 token count sanity — run {run_id}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_latency_and_agreement(db_path: str, run_id: str, out_path: str | Path) -> Path:
    conn = open_ro(db_path)
    try:
        lat = latency_stats_by_budget(conn, run_id)
        agr = uct_top3_agreement_by_budget(conn, run_id)
    finally:
        conn.close()
    budgets = sorted(set(lat.keys()) | set(agr.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(budgets, [lat.get(b, {}).get("mean_ms", 0) for b in budgets], "o-")
    axes[0].set_xscale("symlog")
    axes[0].set_xticks(budgets)
    axes[0].set_xticklabels([str(b) for b in budgets])
    axes[0].set_xlabel("Budget B")
    axes[0].set_ylabel("Mean LLM-turn latency (ms)")
    axes[0].set_title("Latency vs B")
    axes[0].grid(alpha=0.3)

    axes[1].plot(budgets, [agr.get(b, {}).get("agreement", 0) for b in budgets], "o-")
    axes[1].axhline(3 / 30, color="gray", ls="--", lw=1, label="random ~10%")  # rough
    axes[1].set_xscale("symlog")
    axes[1].set_xticks(budgets)
    axes[1].set_xticklabels([str(b) for b in budgets])
    axes[1].set_xlabel("Budget B")
    axes[1].set_ylabel("Fraction of LLM moves in UCT top-3")
    axes[1].set_title("Move quality vs B")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Latency & move quality — run {run_id}")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
