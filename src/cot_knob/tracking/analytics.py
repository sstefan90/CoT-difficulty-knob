"""Read-side analytics over the SQLite store.

Kept thin; the canonical queries live as functions here so plotting code
and notebooks stay declarative.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import numpy as np


def open_ro(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def trials_for_run(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    cur = conn.cursor()
    return list(cur.execute(
        "SELECT * FROM trials WHERE run_id=? ORDER BY cell_index",
        (run_id,),
    ))


def win_rate_by_budget(conn: sqlite3.Connection, run_id: str) -> dict[int, dict]:
    """Return { B: {n, wins, win_rate, wilson_low, wilson_high} }."""
    rows = trials_for_run(conn, run_id)
    by_b: dict[int, list[int]] = {}
    for r in rows:
        cond = json.loads(r["condition_json"])
        b = int(cond.get("budget", -1))
        win = 1 if r["winner"] == "llm" else 0
        by_b.setdefault(b, []).append(win)

    out: dict[int, dict] = {}
    for b, vals in sorted(by_b.items()):
        n = len(vals)
        w = sum(vals)
        p = w / n if n else 0.0
        lo, hi = _wilson(p, n)
        out[b] = {"n": n, "wins": w, "win_rate": p, "wilson_low": lo, "wilson_high": hi}
    return out


def reasoning_token_stats_by_budget(conn: sqlite3.Connection, run_id: str) -> dict[int, dict]:
    """Sanity check: did we actually get B output tokens? Returns mean/median per B."""
    cur = conn.cursor()
    rows = list(cur.execute("""
        SELECT t.condition_json AS cond, mc.n_output_tokens AS n
          FROM model_calls mc
          JOIN trials t ON mc.trial_id = t.trial_id
         WHERE t.run_id = ? AND mc.role = 'reason'
    """, (run_id,)))
    by_b: dict[int, list[int]] = {}
    for r in rows:
        b = int(json.loads(r["cond"]).get("budget", -1))
        by_b.setdefault(b, []).append(int(r["n"]))
    return {
        b: {
            "n_calls": len(vals),
            "mean_tokens_out": float(np.mean(vals)) if vals else 0.0,
            "median_tokens_out": float(np.median(vals)) if vals else 0.0,
            "p95_tokens_out": float(np.percentile(vals, 95)) if vals else 0.0,
        }
        for b, vals in sorted(by_b.items())
    }


def latency_stats_by_budget(conn: sqlite3.Connection, run_id: str) -> dict[int, dict]:
    cur = conn.cursor()
    rows = list(cur.execute("""
        SELECT t.condition_json AS cond, tu.latency_ms_total AS l
          FROM turns tu
          JOIN trials t ON tu.trial_id = t.trial_id
         WHERE t.run_id = ? AND tu.agent_kind = 'llm'
    """, (run_id,)))
    by_b: dict[int, list[float]] = {}
    for r in rows:
        b = int(json.loads(r["cond"]).get("budget", -1))
        by_b.setdefault(b, []).append(float(r["l"]))
    return {
        b: {"n_turns": len(vals), "mean_ms": mean(vals) if vals else 0.0}
        for b, vals in sorted(by_b.items())
    }


def uct_top3_agreement_by_budget(conn: sqlite3.Connection, run_id: str) -> dict[int, dict]:
    """move_quality is set per-LLM-turn. Aggregate across budgets."""
    cur = conn.cursor()
    rows = list(cur.execute("""
        SELECT t.condition_json AS cond, tu.move_quality AS mq
          FROM turns tu
          JOIN trials t ON tu.trial_id = t.trial_id
         WHERE t.run_id = ? AND tu.agent_kind = 'llm' AND tu.move_quality IS NOT NULL
    """, (run_id,)))
    by_b: dict[int, list[int]] = {}
    for r in rows:
        b = int(json.loads(r["cond"]).get("budget", -1))
        by_b.setdefault(b, []).append(int(r["mq"]))
    return {
        b: {
            "n": len(vals),
            "agreement": (sum(vals) / len(vals)) if vals else 0.0,
        }
        for b, vals in sorted(by_b.items())
    }


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; reasonable even at tiny N."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: Iterable[int | float], n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))
