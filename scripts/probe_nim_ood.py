"""OOD and memorization probe for Nim.

Two sub-probes:

─────────────────────────────────────────────────────────────────────────────
PROBE 1 — LEGALITY (expected: HIGH — Nim moves are trivial arithmetic)
─────────────────────────────────────────────────────────────────────────────
  Show pile state WITHOUT listing legal moves.
  Ask model for one move; extract (pile, count) and check legality.
  Budget sweep: B ∈ {0, 64, 256, 512, 1024}.

  Nim legality is trivially computable: "take N from pile X" is legal iff
  pile X has ≥ N stones. We expect high legal rates even at B=0.

─────────────────────────────────────────────────────────────────────────────
PROBE 2 — STRATEGY / MEMORIZATION (the interesting one)
─────────────────────────────────────────────────────────────────────────────
  Show positions where nim_sum ≠ 0 (current player can win with correct play).
  Ask model for its chosen move; check if it is nim-optimal.
  Use both canonical pile sizes ([3,5,7]) and non-canonical ([4,6,11], ...)
  to separate memorisation from genuine nim-sum reasoning.

  B=0 near-optimal rate → model has memorised the strategy.
  B>0 improving rate    → model computes nim-sum during reasoning.

Architecture:
  --no-think (default, Llama): single generate call with think=False.
    The response field contains the full output (reasoning + move).
    B=0 → 128 tokens, no reasoning preamble.
    B>0 → B tokens, reasoning preamble included.
  --think (R1): single generate call with think=True.
    num_predict=B controls the thinking field.
    The response field contains the final move (if thinking completed).
    No Pass-2 commit call — we extract from the merged output only.

Output is saved to data/probes/nim_ood.jsonl (or --out path).

Usage
─────
    uv run python -u scripts/probe_nim_ood.py --model llama3.1:8b --n-positions 5
    uv run python -u scripts/probe_nim_ood.py --model deepseek-r1:7b --think --n-positions 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cot_knob.agents.nim_optimal import _nim_winning_moves
from cot_knob.games.nim import NimState

OUTPUT_DIR = ROOT / "data" / "probes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "nim_ood.jsonl"

BUDGETS = [0, 64, 256, 512, 1024]

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _ollama_generate(
    prompt: str,
    *,
    think: bool = False,
    num_predict: int = 256,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    import urllib.request

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
        "think": think,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _get_text(data: dict[str, Any], *, think_mode: bool) -> str:
    """Extract model output text from Ollama response.

    think_mode=False (Llama): only the response field is used.
    think_mode=True  (R1):    merge thinking + response.
    """
    response = (data.get("response") or "").strip()
    if not think_mode:
        return response
    # R1: merge thinking + response for full context
    thinking = (data.get("thinking") or "").strip()
    if thinking and response:
        return f"{thinking}\n\n{response}"
    return thinking or response


# ── Move extraction ───────────────────────────────────────────────────────────

def _extract_nim_move(text: str) -> tuple[int, int] | None:
    """Extract (pile_letter_index, stones) from free-form model output.

    Returns None if no parseable move is found.
    Uses the LAST matching pattern found (model often restates at end).

    Patterns ordered most-specific first:
      1. "MOVE: pile=X take=N"  (structured tag — most reliable)
      2. "take/remove N [stones] from [pile] X"
      3. "N [stones] from [pile] X"
      4. "from [pile] X ... N"
      5. "pile X ... N"
    """
    text_upper = text.upper()
    patterns = [
        # Structured MOVE tag (primary for Llama)
        r"MOVE:\s*PILE=([A-Z])\s+TAKE=(\d+)",
        # "take/remove N [stones] from [pile] X"
        r"(?:TAKE|REMOVE)\s+(\d+)\s+(?:STONES?\s+)?FROM\s+(?:PILE\s+)?([A-Z])\b",
        # "N [stones] from [pile] X"
        r"(\d+)\s+(?:STONES?\s+)?FROM\s+(?:PILE\s+)?([A-Z])\b",
        # "from [pile] X ... N"
        r"FROM\s+(?:PILE\s+)?([A-Z])\b[^0-9A-Z]*(\d+)",
        # "pile X ... N"
        r"\bPILE\s+([A-Z])\b[^0-9]*(\d+)",
    ]
    matches: list[tuple[int, int]] = []
    for pat in patterns:
        for m in re.finditer(pat, text_upper):
            g = m.groups()
            try:
                if g[0].isdigit():
                    stones, pile_ch = int(g[0]), g[1]
                else:
                    pile_ch, stones = g[0], int(g[1])
                pile_idx = ord(pile_ch) - ord("A")
                if 0 <= pile_idx <= 25 and stones >= 1:
                    matches.append((pile_idx, stones))
            except (ValueError, IndexError):
                continue
    if matches:
        return matches[-1]  # last match wins
    return None


# ── Position generators ───────────────────────────────────────────────────────

CANONICAL_CONFIGS: list[tuple[int, ...]] = [
    (3, 5, 7),
    (1, 2, 3),
    (1, 3, 5, 7),
    (2, 4, 6),
]

NON_CANONICAL_CONFIGS: list[tuple[int, ...]] = [
    (4, 6, 11),
    (2, 8, 13),
    (3, 7, 12),
    (5, 9, 14),
    (6, 10, 15),
]


def _generate_positions(
    n: int = 5,
    *,
    non_canonical_only: bool = False,
    require_winning: bool = False,
) -> list[NimState]:
    import random
    from functools import reduce
    from operator import xor

    rng = random.Random(42)
    positions: list[NimState] = []
    configs = NON_CANONICAL_CONFIGS if non_canonical_only else (CANONICAL_CONFIGS + NON_CANONICAL_CONFIGS)

    attempts = 0
    while len(positions) < n and attempts < 10_000:
        attempts += 1
        config = rng.choice(configs)
        piles = tuple(rng.randint(1, p) for p in config)
        if all(p == 0 for p in piles):
            continue
        nim_sum = reduce(xor, piles, 0)
        if require_winning and nim_sum == 0:
            continue
        positions.append(NimState(piles))

    return positions


# ── Probe 1: Legality ─────────────────────────────────────────────────────────

LEGALITY_SYSTEM = (
    "You are playing Nim. On your turn remove at least 1 stone from exactly "
    "one non-empty pile. The player who takes the last stone wins.\n"
    "End your response with exactly: MOVE: pile=X take=N"
)

# B=0: no reasoning preamble — model answers from prior directly
LEGALITY_USER_DIRECT = """{board}

Output your move:
MOVE: pile="""

# B>0: model reasons then commits
LEGALITY_USER_REASON = """{board}

Reason briefly about a legal move, then end with:
MOVE: pile=X take=N"""


async def run_legality_probe(
    positions: list[NimState],
    budget: int,
    *,
    out_records: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    think_mode: bool = False,
    on_record=None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pos_idx, state in enumerate(positions):
        board_text = state.render_text(show_legal=False)

        t0 = time.perf_counter()
        try:
            if budget == 0:
                # No reasoning budget: direct answer from prior.
                # Prefix "pile=" to nudge the model to complete the MOVE tag.
                user_msg = LEGALITY_USER_DIRECT.format(board=board_text)
                full_prompt = f"{LEGALITY_SYSTEM}\n\n{user_msg}"
                data = _ollama_generate(full_prompt, think=think_mode, num_predict=16, model=model)
                # Reconstruct the MOVE tag: prompt ended with "pile=" and model completes it.
                completion = _get_text(data, think_mode=think_mode)
                raw = f"MOVE: pile={completion}"
            else:
                user_msg = LEGALITY_USER_REASON.format(board=board_text)
                full_prompt = f"{LEGALITY_SYSTEM}\n\n{user_msg}"
                data = _ollama_generate(full_prompt, think=think_mode, num_predict=budget, model=model)
                raw = _get_text(data, think_mode=think_mode)
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        extracted = _extract_nim_move(raw)
        if extracted is None:
            outcome = "no_parse"
            is_legal = False
            move_str = None
        else:
            pile_idx, stones = extracted
            move_str = f"take {stones} from {chr(ord('A') + pile_idx)}"
            if 0 <= pile_idx < len(state.piles) and 1 <= stones <= state.piles[pile_idx]:
                is_legal = True
                outcome = "legal"
            else:
                is_legal = False
                outcome = "illegal"

        record: dict[str, Any] = {
            "probe": "legality",
            "model": model,
            "think_mode": think_mode,
            "budget": budget,
            "pos_idx": pos_idx,
            "piles": list(state.piles),
            "outcome": outcome,
            "is_legal": is_legal,
            "extracted": list(extracted) if extracted else None,
            "move_str": move_str,
            "raw_output": raw[:600],
            "latency_ms": round(latency_ms, 1),
        }
        out_records.append(record)
        results.append(record)
        if on_record is not None:
            on_record(record)
        status = "✓" if is_legal else ("?" if outcome == "no_parse" else "✗")
        print(
            f"  [legality] B={budget:>4} pos={pos_idx} piles={list(state.piles)} "
            f"→ {outcome} {status}",
            flush=True,
        )

    n_legal = sum(1 for r in results if r["is_legal"])
    print(
        f"  [legality] B={budget} — legal rate: {n_legal}/{len(positions)} "
        f"= {n_legal/max(1,len(positions)):.0%}",
        flush=True,
    )
    return {"budget": budget, "n_positions": len(positions), "n_legal": n_legal}


# ── Probe 2: Strategy / memorization ─────────────────────────────────────────

STRATEGY_SYSTEM = (
    "You are playing Nim. Remove at least 1 stone from exactly one non-empty pile. "
    "The player who takes the last stone wins. Apply nim-sum strategy.\n"
    "End your response with exactly: MOVE: pile=X take=N"
)

STRATEGY_USER_DIRECT = """{board}

Output your optimal move:
MOVE: pile="""

STRATEGY_USER_REASON = """{board}

Compute the nim-sum and determine the winning move. End with:
MOVE: pile=X take=N"""


async def run_strategy_probe(
    positions: list[NimState],
    budget: int,
    *,
    out_records: list[dict[str, Any]],
    tag: str = "canonical",
    model: str = DEFAULT_MODEL,
    think_mode: bool = False,
    on_record=None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pos_idx, state in enumerate(positions):
        board_text = state.render_text(show_legal=False)

        winning = _nim_winning_moves(state.piles)
        winning_strs = {
            f"take {s} from {chr(ord('A') + i)}" for i, s in winning
        }

        t0 = time.perf_counter()
        try:
            if budget == 0:
                user_msg = STRATEGY_USER_DIRECT.format(board=board_text)
                full_prompt = f"{STRATEGY_SYSTEM}\n\n{user_msg}"
                data = _ollama_generate(full_prompt, think=think_mode, num_predict=16, model=model)
                completion = _get_text(data, think_mode=think_mode)
                raw = f"MOVE: pile={completion}"
            else:
                user_msg = STRATEGY_USER_REASON.format(board=board_text)
                full_prompt = f"{STRATEGY_SYSTEM}\n\n{user_msg}"
                data = _ollama_generate(full_prompt, think=think_mode, num_predict=budget, model=model)
                raw = _get_text(data, think_mode=think_mode)
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        extracted = _extract_nim_move(raw)
        if extracted is None:
            outcome = "no_parse"
            is_optimal = False
            move_str = None
        else:
            pile_idx, stones = extracted
            move_str = f"take {stones} from {chr(ord('A') + pile_idx)}"
            is_optimal = move_str in winning_strs
            outcome = "optimal" if is_optimal else "suboptimal"

        record: dict[str, Any] = {
            "probe": "strategy",
            "tag": tag,
            "model": model,
            "think_mode": think_mode,
            "budget": budget,
            "pos_idx": pos_idx,
            "piles": list(state.piles),
            "nim_sum": state.to_serializable()["nim_sum"],
            "winning_moves": list(winning_strs),
            "outcome": outcome,
            "is_optimal": is_optimal,
            "extracted": list(extracted) if extracted else None,
            "move_str": move_str,
            "raw_output": raw[:800],
            "latency_ms": round(latency_ms, 1),
        }
        out_records.append(record)
        results.append(record)
        if on_record is not None:
            on_record(record)
        status = "✓" if is_optimal else ("?" if outcome == "no_parse" else "✗")
        print(
            f"  [strategy/{tag}] B={budget:>4} pos={pos_idx} "
            f"piles={list(state.piles)} nim_sum={record['nim_sum']} "
            f"→ {outcome} {status}",
            flush=True,
        )

    n_opt = sum(1 for r in results if r["is_optimal"])
    print(
        f"  [strategy/{tag}] B={budget} — optimal rate: {n_opt}/{len(positions)} "
        f"= {n_opt/max(1,len(positions)):.0%}",
        flush=True,
    )
    return {"budget": budget, "tag": tag, "n_positions": len(positions), "n_optimal": n_opt}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser(description="Nim OOD + strategy probe")
    ap.add_argument("--n-positions", type=int, default=8,
                    help="Positions per budget cell (default: 8)")
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--no-strategy", action="store_true",
                    help="Skip the strategy/memorisation probe")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--think", action="store_true",
                    help="Use think=True (R1 mode). Default is think=False (Llama mode).")
    ap.add_argument("--out", default=str(OUTPUT_FILE), help="Output JSONL path")
    args = ap.parse_args()

    model_name = args.model
    think_mode = args.think
    mode_label = "R1/think" if think_mode else "Llama/no-think"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_fh = open(out_path, "w", encoding="utf-8")  # noqa: SIM115

    def _write_record(rec: dict[str, Any]) -> None:
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    legality_positions = _generate_positions(args.n_positions)
    canonical_positions = [NimState(c) for c in CANONICAL_CONFIGS[:args.n_positions]]
    non_canonical_positions = _generate_positions(
        args.n_positions, non_canonical_only=True, require_winning=True
    )

    print(f"\n{'='*60}", flush=True)
    print(f"Nim OOD Probe — model={model_name}  mode={mode_label}", flush=True)
    print(f"  {len(args.budgets)} budgets × {args.n_positions} positions", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Probe 1: Legality ──────────────────────────────────────────────────
    print("── PROBE 1: LEGALITY ──", flush=True)
    for B in args.budgets:
        row = await run_legality_probe(
            legality_positions, B,
            out_records=records, model=model_name,
            think_mode=think_mode, on_record=_write_record,
        )
        summary_rows.append(row)

    if not args.no_strategy:
        # ── Probe 2a: Strategy (canonical) ────────────────────────────────
        print("\n── PROBE 2a: STRATEGY (canonical piles) ──", flush=True)
        for B in args.budgets:
            row = await run_strategy_probe(
                canonical_positions, B,
                out_records=records, tag="canonical",
                model=model_name, think_mode=think_mode, on_record=_write_record,
            )
            summary_rows.append(row)

        # ── Probe 2b: Strategy (non-canonical) ───────────────────────────
        print("\n── PROBE 2b: STRATEGY (non-canonical piles) ──", flush=True)
        for B in args.budgets:
            row = await run_strategy_probe(
                non_canonical_positions, B,
                out_records=records, tag="non_canonical",
                model=model_name, think_mode=think_mode, on_record=_write_record,
            )
            summary_rows.append(row)

    out_fh.close()
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved → {out_path}", flush=True)
    print(f"Total records: {len(records)}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
