"""OOD and memorization probe for Nim.

Two sub-probes:

─────────────────────────────────────────────────────────────────────────────
PROBE 1 — LEGALITY (expected: HIGH — Nim moves are trivial arithmetic)
─────────────────────────────────────────────────────────────────────────────
  Show pile state WITHOUT listing legal moves.
  Ask model for one move; check legality.
  Budget sweep: B ∈ {0, 64, 256, 512, 1024}.

  With --two-pass (SGLang): Pass 2 always picks from the enumerated legal
  moves, so legality is 100% by construction.  Probe 1 still runs to verify
  no errors, measure latency, and confirm the cache hit behaviour.

─────────────────────────────────────────────────────────────────────────────
PROBE 2 — STRATEGY / MEMORIZATION (the interesting one)
─────────────────────────────────────────────────────────────────────────────
  Show positions where nim_sum ≠ 0 (current player can win with correct play).
  Ask model for its chosen move; check if it is nim-optimal.
  Use both canonical pile sizes ([3,5,7]) and non-canonical ([4,6,11], ...)
  to separate memorisation from genuine nim-sum reasoning.

  B=0 near-optimal rate → model has memorised the strategy.
  B>0 improving rate    → model computes nim-sum during reasoning.

Backends / architectures
────────────────────────
  --backend ollama          Single-pass, free text, post-hoc parsing.
  --backend sglang          Single-pass unconstrained (same as Ollama path).
  --backend sglang          Two-pass with --two-pass:
    Pass 1: generate(max_tokens=B)  — free reasoning, no regex
    Pass 2: generate_choice(legal_moves)  — prefix-free regex, tight budget,
            always produces a valid legal move (structurally guaranteed)

  The --two-pass architecture exactly mirrors LLMAgent in the game sweeps.
  Use it as the canonical probe for comparing against game sweep results.

Output
──────
  Unconstrained records: raw_output + extracted move.
  Two-pass records: pass1_text + pass1_finish + pass2_choice + latency/tokens.

Usage
─────
    # Ollama baseline
    uv run python -u scripts/probe_nim_ood.py --backend ollama --n-positions 8

    # SGLang unconstrained (fair comparison baseline)
    uv run python -u scripts/probe_nim_ood.py --backend sglang --n-positions 8

    # SGLang two-pass (matches game sweep architecture)
    uv run python -u scripts/probe_nim_ood.py --backend sglang --two-pass --n-positions 8
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
SGLANG_BASE = "http://localhost:30000"
DEFAULT_MODEL_OLLAMA = "llama3.1:8b"
DEFAULT_MODEL_SGLANG = "default"  # SGLang serves one model; "name:tag" breaks as LoRA ref


# ── Backend helpers ───────────────────────────────────────────────────────────

def _ollama_generate(
    prompt: str,
    *,
    system: str | None = None,
    think: bool = False,
    num_predict: int = 256,
    model: str = DEFAULT_MODEL_OLLAMA,
    base_url: str = OLLAMA_BASE,
) -> str:
    """Call Ollama /api/generate and return the output text."""
    import urllib.request

    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
        "think": think,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read())
    response = (raw.get("response") or "").strip()
    if think:
        thinking = (raw.get("thinking") or "").strip()
        return f"{thinking}\n\n{response}" if thinking and response else thinking or response
    return response


def _sglang_generate(
    prompt: str,
    *,
    system: str | None = None,
    num_predict: int = 256,
    model: str = "default",
    base_url: str = SGLANG_BASE,
    regex: str | None = None,
) -> str:
    """Call SGLang /v1/chat/completions and return the output text only."""
    return _sglang_call(
        prompt, system=system, num_predict=num_predict,
        model=model, base_url=base_url, regex=regex,
    )["text"]


def _sglang_call(
    prompt: str,
    *,
    system: str | None = None,
    num_predict: int = 256,
    model: str = "default",
    base_url: str = SGLANG_BASE,
    regex: str | None = None,
) -> dict[str, Any]:
    """Call SGLang and return full response: text, finish_reason, token counts."""
    import urllib.request

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": int(num_predict),
        "temperature": 0.0,
        "stream": False,
    }
    if regex is not None:
        payload["sampling_params"] = {"regex": regex}

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read())

    choice = raw["choices"][0]
    usage = raw.get("usage") or {}
    # SGLang may return prompt_tokens_details.cached_tokens if prefix cache hit
    ptd = usage.get("prompt_tokens_details") or {}
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason") or "stop",
        "n_input_tokens": int(usage.get("prompt_tokens", 0)),
        "n_output_tokens": int(usage.get("completion_tokens", 0)),
        "cached_tokens": ptd.get("cached_tokens"),  # None if SGLang doesn't expose it
    }


def _generate(
    prompt: str,
    *,
    system: str | None = None,
    num_predict: int,
    backend: str,
    model: str,
    base_url: str,
    think: bool = False,
    regex: str | None = None,
) -> str:
    """Dispatch to the appropriate backend."""
    if backend == "sglang":
        return _sglang_generate(
            prompt, system=system, num_predict=num_predict,
            model=model, base_url=base_url, regex=regex,
        )
    # default: ollama
    return _ollama_generate(
        prompt, system=system, think=think, num_predict=num_predict,
        model=model, base_url=base_url,
    )


# ── Two-pass helper (SGLang only) ─────────────────────────────────────────────

# Pass 2 system prompt — mirrors nim_prompts.SYSTEM_PROMPT_PASS2
SELECT_SYSTEM = (
    "Same Nim rules apply. "
    "Output exactly ONE move from the list you are given. "
    "No explanation, no punctuation — just the move string (e.g. take 2 from A)."
)

# Pass 2 user prompt — mirrors nim_prompts.SELECT_TEMPLATE
SELECT_USER_TEMPLATE = """{prior}

Choose exactly one of: {choices}"""


def _two_pass_probe(
    state: NimState,
    budget: int,
    *,
    reason_system: str,
    reason_user: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    """Run the two-pass probe mirroring LLMAgent._choose_nim.

    Pass 1: free reasoning, max_tokens=budget (skip entirely when budget=0).
    Pass 2: generate_choice from enumerated legal moves — always produces a
            valid legal move; mirrors generate_choice() in SGLangClient.

    Returns a dict with all telemetry fields ready for the probe record.
    """
    legal_moves = state.legal_moves()
    choices = [state.move_to_str(m) for m in legal_moves]
    choice_regex = "(" + "|".join(re.escape(c) for c in choices) + ")"
    max_choice_len = max(len(c) for c in choices) if choices else 8

    # ── Pass 1: free reasoning ────────────────────────────────────────────────
    p1_t0 = time.perf_counter()
    if budget > 0:
        p1 = _sglang_call(
            reason_user, system=reason_system, num_predict=budget,
            model=model, base_url=base_url,
        )
        pass1_text = p1["text"]
        pass1_finish = p1["finish_reason"]
        pass1_tokens_in = p1["n_input_tokens"]
        pass1_tokens_out = p1["n_output_tokens"]
    else:
        pass1_text = ""
        pass1_finish = "skipped"
        pass1_tokens_in = 0
        pass1_tokens_out = 0
    pass1_latency_ms = round((time.perf_counter() - p1_t0) * 1000, 1)

    # ── Pass 2: constrained choice ────────────────────────────────────────────
    # Include reasoning in context exactly as LLMAgent does.
    select_prior = (
        f"{reason_user}\n\n"
        f"<reasoning>\n{pass1_text}\n</reasoning>"
    )
    select_user = SELECT_USER_TEMPLATE.format(
        prior=select_prior, choices=", ".join(choices)
    )

    p2_t0 = time.perf_counter()
    p2 = _sglang_call(
        select_user, system=SELECT_SYSTEM,
        num_predict=max_choice_len + 4,
        model=model, base_url=base_url, regex=choice_regex,
    )
    pass2_latency_ms = round((time.perf_counter() - p2_t0) * 1000, 1)

    chosen = p2["text"].strip()
    # Exact match; fallback to substring (should be rare with regex enforcement)
    if chosen not in choices:
        chosen = next((c for c in choices if c in chosen), choices[0])

    return {
        "pass1_text": pass1_text[:800],
        "pass1_finish": pass1_finish,
        "pass1_tokens_in": pass1_tokens_in,
        "pass1_tokens_out": pass1_tokens_out,
        "pass1_latency_ms": pass1_latency_ms,
        "pass2_choice": chosen,
        "pass2_prompt_tokens": p2["n_input_tokens"],
        "pass2_cached_tokens": p2["cached_tokens"],
        "pass2_latency_ms": pass2_latency_ms,
    }


# ── Move extraction (single-pass only) ───────────────────────────────────────

def _extract_nim_move(text: str) -> tuple[int, int] | None:
    """Extract (pile_letter_index, stones) from free-form model output.

    Returns None if no parseable move is found.
    Uses the last match BY TEXT POSITION across all patterns, so the MOVE tag
    at the end of the response always wins over pile labels in reasoning.

    Patterns ordered most-specific first (priority used to break ties):
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
        # "pile X ... N"  (broad fallback — least specific)
        r"\bPILE\s+([A-Z])\b[^0-9]*(\d+)",
    ]
    # Track (text_position, pattern_priority, pile_idx, stones).
    # Returning the last-by-position match ensures the MOVE tag at the end
    # of a response beats binary representations in the reasoning prefix.
    candidates: list[tuple[int, int, int, int]] = []
    for priority, pat in enumerate(patterns):
        for m in re.finditer(pat, text_upper):
            g = m.groups()
            try:
                if g[0].isdigit():
                    stones, pile_ch = int(g[0]), g[1]
                else:
                    pile_ch, stones = g[0], int(g[1])
                pile_idx = ord(pile_ch) - ord("A")
                if 0 <= pile_idx <= 25 and stones >= 1:
                    candidates.append((m.start(), priority, pile_idx, stones))
            except (ValueError, IndexError):
                continue
    if not candidates:
        return None
    # Primary sort: latest text position. Tie-break: lowest priority index (most specific).
    best = max(candidates, key=lambda c: (c[0], -c[1]))
    return (best[2], best[3])


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
    model: str = DEFAULT_MODEL_SGLANG,
    backend: str = "ollama",
    base_url: str = OLLAMA_BASE,
    think_mode: bool = False,
    two_pass: bool = False,
    on_record=None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pos_idx, state in enumerate(positions):
        board_text = state.render_text(show_legal=False)
        reason_user = LEGALITY_USER_REASON.format(board=board_text)

        t0 = time.perf_counter()
        tp: dict[str, Any] = {}
        try:
            if two_pass:
                # ── Two-pass: Pass 2 always picks a legal move ─────────────
                tp = _two_pass_probe(
                    state, budget,
                    reason_system=LEGALITY_SYSTEM,
                    reason_user=reason_user,
                    model=model, base_url=base_url,
                )
                raw = tp["pass2_choice"]
            elif budget == 0 and backend != "sglang":
                # Ollama B=0: direct prefix completion
                user_msg = LEGALITY_USER_DIRECT.format(board=board_text)
                completion = _generate(
                    user_msg, system=None, num_predict=16,
                    backend=backend, model=model, base_url=base_url, think=think_mode,
                )
                raw = f"MOVE: pile={completion}"
            else:
                user_msg = reason_user
                n_tok = max(budget, 32) if budget == 0 else budget
                raw = _generate(
                    user_msg, system=LEGALITY_SYSTEM, num_predict=n_tok,
                    backend=backend, model=model, base_url=base_url,
                    think=think_mode,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  [legality] ERROR pos={pos_idx}: {e}", flush=True)
            raw = f"ERROR: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        # For two-pass: pass2_choice is always legal (by construction).
        # For single-pass: parse the raw output.
        if two_pass:
            chosen = raw  # pass2_choice is already a "take N from X" string
            # Parse the choice string back to (pile_idx, stones)
            m = re.match(r"take (\d+) from ([A-Z])", chosen, re.I)
            if m:
                stones = int(m.group(1))
                pile_idx = ord(m.group(2).upper()) - ord("A")
                is_legal = 0 <= pile_idx < len(state.piles) and 1 <= stones <= state.piles[pile_idx]
                outcome = "legal" if is_legal else "illegal"
                move_str = chosen
            else:
                is_legal = False
                outcome = "no_parse"
                move_str = None
        else:
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
            "backend": backend,
            "two_pass": two_pass,
            "think_mode": think_mode,
            "budget": budget,
            "pos_idx": pos_idx,
            "piles": list(state.piles),
            "outcome": outcome,
            "is_legal": is_legal,
            "move_str": move_str,
            "latency_ms": round(latency_ms, 1),
        }
        if two_pass:
            record.update({
                "pass1_text": tp.get("pass1_text", ""),
                "pass1_finish": tp.get("pass1_finish", ""),
                "pass1_tokens_in": tp.get("pass1_tokens_in", 0),
                "pass1_tokens_out": tp.get("pass1_tokens_out", 0),
                "pass1_latency_ms": tp.get("pass1_latency_ms", 0.0),
                "pass2_choice": tp.get("pass2_choice", ""),
                "pass2_prompt_tokens": tp.get("pass2_prompt_tokens", 0),
                "pass2_cached_tokens": tp.get("pass2_cached_tokens"),
                "pass2_latency_ms": tp.get("pass2_latency_ms", 0.0),
            })
        else:
            record["raw_output"] = raw[:600]

        out_records.append(record)
        results.append(record)
        if on_record is not None:
            on_record(record)
        status = "✓" if is_legal else ("?" if outcome == "no_parse" else "✗")
        print(
            f"  [legality] B={budget:>4} pos={pos_idx} piles={list(state.piles)} "
            f"→ {outcome} {status}  move={move_str}",
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
    model: str = DEFAULT_MODEL_SGLANG,
    backend: str = "ollama",
    base_url: str = OLLAMA_BASE,
    think_mode: bool = False,
    two_pass: bool = False,
    on_record=None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pos_idx, state in enumerate(positions):
        board_text = state.render_text(show_legal=False)
        reason_user = STRATEGY_USER_REASON.format(board=board_text)

        winning = _nim_winning_moves(state.piles)
        winning_strs = {
            f"take {s} from {chr(ord('A') + i)}" for i, s in winning
        }

        t0 = time.perf_counter()
        tp: dict[str, Any] = {}
        try:
            if two_pass:
                tp = _two_pass_probe(
                    state, budget,
                    reason_system=STRATEGY_SYSTEM,
                    reason_user=reason_user,
                    model=model, base_url=base_url,
                )
                raw = tp["pass2_choice"]
            elif budget == 0 and backend != "sglang":
                user_msg = STRATEGY_USER_DIRECT.format(board=board_text)
                completion = _generate(
                    user_msg, system=None, num_predict=16,
                    backend=backend, model=model, base_url=base_url, think=think_mode,
                )
                raw = f"MOVE: pile={completion}"
            else:
                n_tok = max(budget, 32) if budget == 0 else budget
                raw = _generate(
                    reason_user, system=STRATEGY_SYSTEM, num_predict=n_tok,
                    backend=backend, model=model, base_url=base_url,
                    think=think_mode,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  [strategy/{tag}] ERROR pos={pos_idx}: {e}", flush=True)
            raw = f"ERROR: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        if two_pass:
            move_str = raw  # pass2_choice is a legal "take N from X" string
            is_optimal = move_str in winning_strs
            outcome = "optimal" if is_optimal else "suboptimal"
        else:
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
            "backend": backend,
            "two_pass": two_pass,
            "think_mode": think_mode,
            "budget": budget,
            "pos_idx": pos_idx,
            "piles": list(state.piles),
            "nim_sum": state.to_serializable()["nim_sum"],
            "winning_moves": list(winning_strs),
            "outcome": outcome,
            "is_optimal": is_optimal,
            "move_str": move_str,
            "latency_ms": round(latency_ms, 1),
        }
        if two_pass:
            record.update({
                "pass1_text": tp.get("pass1_text", ""),
                "pass1_finish": tp.get("pass1_finish", ""),
                "pass1_tokens_in": tp.get("pass1_tokens_in", 0),
                "pass1_tokens_out": tp.get("pass1_tokens_out", 0),
                "pass1_latency_ms": tp.get("pass1_latency_ms", 0.0),
                "pass2_choice": tp.get("pass2_choice", ""),
                "pass2_prompt_tokens": tp.get("pass2_prompt_tokens", 0),
                "pass2_cached_tokens": tp.get("pass2_cached_tokens"),
                "pass2_latency_ms": tp.get("pass2_latency_ms", 0.0),
            })
        else:
            record["raw_output"] = raw[:800]

        out_records.append(record)
        results.append(record)
        if on_record is not None:
            on_record(record)
        status = "✓" if is_optimal else ("?" if outcome == "no_parse" else "✗")
        print(
            f"  [strategy/{tag}] B={budget:>4} pos={pos_idx} "
            f"piles={list(state.piles)} nim_sum={record['nim_sum']} "
            f"→ {outcome} {status}  move={move_str}",
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
    ap.add_argument("--no-legality", action="store_true",
                    help="Skip Probe 1 (legality).")
    ap.add_argument("--no-strategy", action="store_true",
                    help="Skip Probe 2 (strategy/memorisation).")
    ap.add_argument("--model", default=None,
                    help="Model name for API requests. Defaults: 'default' for SGLang, "
                         "'llama3.1:8b' for Ollama.")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "sglang"],
                    help="Inference backend (default: ollama).")
    ap.add_argument("--ollama-url", default=OLLAMA_BASE, help="Ollama base URL.")
    ap.add_argument("--sglang-url", default=SGLANG_BASE, help="SGLang base URL.")
    ap.add_argument("--two-pass", action="store_true",
                    help="SGLang only: use two-pass architecture (Pass 1 = free reasoning, "
                         "Pass 2 = generate_choice over legal moves). Matches game sweep "
                         "architecture exactly. Legality is 100%% by construction.")
    # Keep --constrained as a deprecated alias for --two-pass
    ap.add_argument("--constrained", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--think", action="store_true",
                    help="Ollama only: use think=True (R1 mode). Default: think=False.")
    ap.add_argument("--out", default=str(OUTPUT_FILE), help="Output JSONL path.")
    args = ap.parse_args()

    # --constrained is a legacy alias for --two-pass
    two_pass = args.two_pass or args.constrained

    if two_pass and args.backend != "sglang":
        ap.error("--two-pass requires --backend sglang")

    # Auto-select model name based on backend if not explicitly provided.
    if args.model is not None:
        model_name = args.model
    elif args.backend == "sglang":
        model_name = DEFAULT_MODEL_SGLANG
    else:
        model_name = DEFAULT_MODEL_OLLAMA
    think_mode = args.think
    backend = args.backend
    base_url = args.sglang_url if backend == "sglang" else args.ollama_url

    arch_label = "two-pass" if two_pass else "single-pass"
    mode_label = f"{backend} {arch_label}"
    if backend == "ollama":
        mode_label += " R1/think" if think_mode else " Llama/no-think"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(out_path, "a", encoding="utf-8")  # append — multi-run comparison

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
    print(f"Nim OOD Probe — model={model_name}  backend={mode_label}", flush=True)
    print(f"  {len(args.budgets)} budgets × {args.n_positions} positions", flush=True)
    if two_pass:
        print("  ⚡ two-pass ON — Pass1=free reasoning  Pass2=generate_choice", flush=True)
        print("     legality 100%% by construction; pass1_text + pass2_choice logged", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Probe 1: Legality ──────────────────────────────────────────────────
    if not args.no_legality:
        print("── PROBE 1: LEGALITY ──", flush=True)
        for B in args.budgets:
            row = await run_legality_probe(
                legality_positions, B,
                out_records=records, model=model_name,
                backend=backend, base_url=base_url,
                think_mode=think_mode, two_pass=two_pass,
                on_record=_write_record,
            )
            summary_rows.append(row)
    else:
        print("── PROBE 1: LEGALITY — skipped ──", flush=True)

    if not args.no_strategy:
        # ── Probe 2a: Strategy (canonical) ────────────────────────────────
        print("\n── PROBE 2a: STRATEGY (canonical piles) ──", flush=True)
        for B in args.budgets:
            row = await run_strategy_probe(
                canonical_positions, B,
                out_records=records, tag="canonical",
                model=model_name, backend=backend, base_url=base_url,
                think_mode=think_mode, two_pass=two_pass,
                on_record=_write_record,
            )
            summary_rows.append(row)

        # ── Probe 2b: Strategy (non-canonical) ───────────────────────────
        print("\n── PROBE 2b: STRATEGY (non-canonical piles) ──", flush=True)
        for B in args.budgets:
            row = await run_strategy_probe(
                non_canonical_positions, B,
                out_records=records, tag="non_canonical",
                model=model_name, backend=backend, base_url=base_url,
                think_mode=think_mode, two_pass=two_pass,
                on_record=_write_record,
            )
            summary_rows.append(row)

    out_fh.close()
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved → {out_path}  (append mode — previous runs preserved)", flush=True)
    print(f"Total records this run: {len(records)}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    if summary_rows:
        print(f"\n{'Probe':<12} {'Tag':<15} {'B':>5}  {'n':>4}  {'rate':>6}", flush=True)
        print("-" * 50, flush=True)
        for r in summary_rows:
            if "n_legal" in r:
                rate = r["n_legal"] / max(1, r["n_positions"])
                print(f"  {'legality':<10} {'':<15} {r['budget']:>5}  {r['n_positions']:>4}  {rate:>6.0%}", flush=True)
            else:
                rate = r["n_optimal"] / max(1, r["n_positions"])
                print(f"  {'strategy':<10} {r['tag']:<15} {r['budget']:>5}  {r['n_positions']:>4}  {rate:>6.0%}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
