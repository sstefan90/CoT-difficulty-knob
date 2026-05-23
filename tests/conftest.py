"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
THIRD_PARTY = ROOT / "third_party"

for p in (str(SRC), str(THIRD_PARTY)):
    if p not in sys.path:
        sys.path.insert(0, p)
