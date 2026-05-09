"""One-off script to rename existing hex-ID run directories to human-readable
names by querying the SQLite database.

Old format:  data/runs/run_c80c54b24df6/
New format:  data/runs/nim_diag_a_nimsum__c80c54b2/

This is safe because the SQLite DB stores run_id values, not file paths.
Analysis scripts that glob data/runs/*/trial_*.jsonl continue to work.

Usage:
    uv run python scripts/rename_run_dirs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "data" / "runs"
DB_PATH = Path(__file__).parent.parent / "data" / "results.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename run directories to human-readable names.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT run_id, name FROM runs ORDER BY started_at").fetchall()
    conn.close()

    id_to_name: dict[str, str] = {r["run_id"]: r["name"] for r in rows}

    renamed = 0
    skipped = 0
    for old_dir in sorted(RUNS_DIR.iterdir()):
        if not old_dir.is_dir():
            continue
        run_id = old_dir.name
        if run_id not in id_to_name:
            print(f"  SKIP  {run_id}  (not in DB)")
            skipped += 1
            continue

        name = id_to_name[run_id]
        short_id = run_id.replace("run_", "")[:8]
        new_name = f"{name}__{short_id}"
        new_dir = RUNS_DIR / new_name

        if old_dir.name == new_name:
            print(f"  OK    {old_dir.name}  (already named correctly)")
            continue

        if new_dir.exists():
            print(f"  SKIP  {old_dir.name}  → {new_name}  (target already exists)")
            skipped += 1
            continue

        print(f"  {'DRY ' if args.dry_run else ''}RENAME  {old_dir.name}  →  {new_name}")
        if not args.dry_run:
            old_dir.rename(new_dir)
            renamed += 1

    print(f"\nDone: {renamed} renamed, {skipped} skipped.")


if __name__ == "__main__":
    main()
