"""One-off post-hoc patch for self-play JSONL files.

In the original sweep runner, the opponent agent was always labeled "uct"
even in self-play matches where the opponent is actually another LLMAgent.
This script renames those labels for a specified run directory.

Usage:
    uv run python scripts/patch_selfplay_labels.py \
        --run-dir data/runs/run_ab4210e9c373 \
        --old-label uct \
        --new-label llm-B1024

The script rewrites each .jsonl file in place, making these substitutions:
  - turn events:     "agent": "uct"    → "agent": "<new-label>"
  - trial_end events: "winner": "uct"  → "winner": "<new-label>"

A .bak file is written alongside each patched file before overwriting.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def patch_trial(path: Path, old_label: str, new_label: str) -> int:
    """Patch one JSONL file. Returns number of lines changed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    changed = 0

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            patched.append(raw)
            continue

        modified = False
        if ev.get("event") == "turn" and ev.get("agent") == old_label:
            ev["agent"] = new_label
            modified = True
        if ev.get("event") == "trial_end" and ev.get("winner") == old_label:
            ev["winner"] = new_label
            modified = True

        patched.append(json.dumps(ev, ensure_ascii=False))
        if modified:
            changed += 1

    if changed:
        backup = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, backup)
        path.write_text("\n".join(patched) + "\n", encoding="utf-8")

    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--old-label", default="uct")
    ap.add_argument("--new-label", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing files")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Directory not found: {run_dir}")

    trials = sorted(run_dir.glob("*.jsonl"))
    print(f"Found {len(trials)} trial files in {run_dir}")
    total_changed = 0

    for trial_path in trials:
        if args.dry_run:
            # Count without writing
            lines = trial_path.read_text().splitlines()
            n = sum(
                1 for raw in lines
                if raw.strip() and (
                    json.loads(raw).get("agent") == args.old_label
                    or json.loads(raw).get("winner") == args.old_label
                ) if raw.strip()
            )
            if n:
                print(f"  [dry-run] {trial_path.name}: {n} lines would change")
            total_changed += n
        else:
            n = patch_trial(trial_path, args.old_label, args.new_label)
            if n:
                print(f"  patched {trial_path.name}: {n} lines changed (backup saved)")
            total_changed += n

    action = "would change" if args.dry_run else "changed"
    print(f"\nDone. Total lines {action}: {total_changed}")


if __name__ == "__main__":
    main()
