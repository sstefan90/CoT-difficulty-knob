"""Append-only JSONL backup for raw traces.

Mirrors the SQLite writes so a corrupted DB never costs us the actual
reasoning text. One file per trial, written line-by-line as events occur.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


class JSONLWriter:
    def __init__(
        self,
        root: str | Path,
        run_id: str,
        *,
        run_name: str | None = None,
    ) -> None:
        # Directory name: "{run_name}__{run_id}" when a name is given,
        # falling back to just the run_id for backward compatibility.
        if run_name:
            short_id = run_id.replace("run_", "")[:8]
            dir_name = f"{run_name}__{short_id}"
        else:
            dir_name = run_id
        self.root = Path(root) / dir_name
        self.root.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}

    def _open(self, trial_id: str):
        if trial_id not in self._files:
            f = (self.root / f"{trial_id}.jsonl").open("a", encoding="utf-8")
            self._files[trial_id] = f
        return self._files[trial_id]

    def write(self, trial_id: str, event: str, payload: dict[str, Any]) -> None:
        rec = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        f = self._open(trial_id)
        f.write(json.dumps(rec, sort_keys=False, default=str) + "\n")
        f.flush()

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        self._files.clear()
