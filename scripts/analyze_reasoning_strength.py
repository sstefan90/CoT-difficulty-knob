"""Aggregate reasoning-trace statistics by CoT budget for a run.

Reads ``model_calls`` rows with ``role='reason'``, joins ``trials`` for
``condition_json->budget``, and reports length / token / latency
distributions plus short exemplar snippets.

Usage::

    uv run python scripts/analyze_reasoning_strength.py <run_id> [--db data/results.db]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def _words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _score_trace(text: str) -> dict:
    """Lightweight heuristics for 'reasoning strength' (exploratory, not ground truth)."""
    t = text.lower()
    return {
        "chars": len(text),
        "words": _words(text),
        "lines": text.count("\n") + (1 if text else 0),
        "mentions_corner": int(bool(re.search(r"\bcorner\b", t))),
        "mentions_edge": int(bool(re.search(r"\bedge\b", t))),
        "mentions_mobility": int(bool(re.search(r"\bmobil", t))),
        "mentions_opponent": int(bool(re.search(r"\bopponent\b|\bthey\b|\bwhite\b", t))),
        "mentions_if": int(t.count(" if ")),
        "question_marks": t.count("?"),
        "numbered_steps": len(re.findall(r"^\s*\d+[\).\s]", text, re.MULTILINE)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--db", default="data/results.db")
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write Markdown report (default: data/runs/<run_id>/reasoning_strength.md)",
    )
    args = p.parse_args()

    db = Path(args.db)
    if not db.is_file():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = list(
        cur.execute(
            """
            SELECT mc.response_text, mc.n_output_tokens, mc.latency_ms, t.condition_json
              FROM model_calls mc
              JOIN trials t ON mc.trial_id = t.trial_id
             WHERE t.run_id = ? AND mc.role = 'reason'
             ORDER BY t.cell_index, mc.rowid
            """,
            (args.run_id,),
        )
    )
    conn.close()

    if not rows:
        raise SystemExit(f"No reason calls for run_id={args.run_id}")

    by_b: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        cond = json.loads(r["condition_json"])
        b = int(cond["budget"])
        text = r["response_text"] or ""
        by_b[b].append(
            {
                "text": text,
                "n_out": int(r["n_output_tokens"] or 0),
                "lat_ms": float(r["latency_ms"] or 0),
                **_score_trace(text),
            }
        )

    budgets = sorted(by_b.keys())
    lines: list[str] = [
        f"# Reasoning strength — `{args.run_id}`",
        "",
        "Heuristic aggregates over Pass-1 (`reason`) calls. *Not* a substitute for hand labels.",
        "",
        "| B | n_calls | mean chars | p50 chars | p90 chars | mean tok_out | mean lat (s) | corner | edge | mobility | opp | if-count | ? | steps |",
        "|---|--------:|-----------:|----------:|----------:|-------------:|-------------:|-------:|-----:|---------:|----:|---------:|--:|------:|",
    ]

    for b in budgets:
        xs = by_b[b]
        chars = [x["chars"] for x in xs]
        toks = [x["n_out"] for x in xs]
        lats = [x["lat_ms"] / 1000.0 for x in xs]

        def pct(p: float) -> float:
            if not chars:
                return 0.0
            s = sorted(chars)
            k = int(round((len(s) - 1) * p))
            return float(s[k])

        n = len(xs)
        mean_c = statistics.mean(chars) if chars else 0
        corner = sum(x["mentions_corner"] for x in xs) / n
        edge = sum(x["mentions_edge"] for x in xs) / n
        mob = sum(x["mentions_mobility"] for x in xs) / n
        opp = sum(x["mentions_opponent"] for x in xs) / n
        ifs = statistics.mean([x["mentions_if"] for x in xs])
        qm = statistics.mean([x["question_marks"] for x in xs])
        steps = statistics.mean([x["numbered_steps"] for x in xs])

        lines.append(
            f"| {b} | {n} | {mean_c:.0f} | {pct(0.5):.0f} | {pct(0.9):.0f} | "
            f"{statistics.mean(toks) if toks else 0:.1f} | {statistics.mean(lats) if lats else 0:.1f} | "
            f"{corner:.2f} | {edge:.2f} | {mob:.2f} | {opp:.2f} | {ifs:.1f} | {qm:.1f} | {steps:.1f} |"
        )

    lines.extend(["", "## Exemplar (longest trace per budget)", ""])
    for b in budgets:
        xs = by_b[b]
        best = max(xs, key=lambda z: z["chars"])
        snippet = best["text"].strip().replace("\r", "")
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n\n… *truncated*"
        lines.append(f"### B = {b}")
        lines.append("")
        lines.append("```")
        lines.append(snippet)
        lines.append("```")
        lines.append("")

    report = "\n".join(lines)
    out = Path(args.out) if args.out else Path("data/runs") / args.run_id / "reasoning_strength.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
