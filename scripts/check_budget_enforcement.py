"""Check that Pass-1 output tokens respect the configured CoT budget B.

For Ollama, ``options.num_predict`` caps **total** generated tokens in one
``/api/generate`` call (thinking + final segment for R1-style models).
We flag rows where ``n_output_tokens > B + slack`` as violations.

Usage::

    uv run python scripts/check_budget_enforcement.py <run_id> [--db data/results.db] [--slack 2]
"""

from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--db", default="data/results.db")
    p.add_argument("--slack", type=int, default=2, help="Allow small server-side overcount.")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = list(
        cur.execute(
            """
            SELECT mc.n_output_tokens, mc.finish_reason, t.condition_json
              FROM model_calls mc
              JOIN trials t ON mc.trial_id = t.trial_id
             WHERE t.run_id = ? AND mc.role = 'reason'
            """,
            (args.run_id,),
        )
    )
    conn.close()
    if not rows:
        print(f"No Pass-1 (reason) calls for run_id={args.run_id}")
        return

    by_b: dict[int, list[tuple[int, str]]] = {}
    violations = 0
    for r in rows:
        b = int(json.loads(r["condition_json"])["budget"])
        n = int(r["n_output_tokens"])
        fr = r["finish_reason"] or ""
        by_b.setdefault(b, []).append((n, fr))
        if n > b + args.slack:
            violations += 1

    print(f"run_id={args.run_id}  reason_calls={len(rows)}  violations(n_out>B+{args.slack})={violations}")
    print()
    for b in sorted(by_b.keys()):
        xs = by_b[b]
        ns = [t[0] for t in xs]
        finishes = [t[1] for t in xs]
        n_stop = sum(1 for f in finishes if f == "stop")
        n_len = sum(1 for f in finishes if f == "length")
        n_other = len(finishes) - n_stop - n_len
        print(
            f"  B={b:>5}  n={len(ns):>4}  "
            f"tok_out min/mean/max={min(ns)}/{sum(ns)/len(ns):.1f}/{max(ns)}  "
            f"finish: stop={n_stop} length={n_len} other={n_other}"
        )
    print()
    if violations:
        print("WARNING: Some rows exceed B+slack — inspect backend token counting vs thinking merge.")
    else:
        print("OK: All Pass-1 token counts are <= B + slack.")


if __name__ == "__main__":
    main()
