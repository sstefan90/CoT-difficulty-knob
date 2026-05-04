"""Probe R1-Distill-7B for out-of-distribution collapse and for memorization
signatures.

From the proposal:

  Risk 3: R1-Distill at B=0 may be out-of-distribution (the model is trained to
  always produce reasoning tokens). Forcing B=0 may collapse output quality.
  Mitigation: test explicitly; if collapse occurs, report curves starting from
  B=64 and note this as a limitation.

  Task 4 (memorization probe): identify ~50 game states where UCT sees multiple
  roughly-equal moves (uncertain positions). At each, sample 10 moves at
  T=0.7, B=0. High consistency on uncertain positions is a memorization
  signature — the model is pattern-matching rather than reasoning.

─────────────────────────────────────────────────────────────────────────────
METHODOLOGY — RAW LEGALITY TEST (the strong OOD check)
─────────────────────────────────────────────────────────────────────────────
Previous version of this script used Pass-2 constrained choice format, which
*guarantees* legal output (legal moves are listed in the prompt + there is a
hard fallback to choices[0]). That gave 100% legal trivially and is not a real
OOD signal.

This version uses a **raw single-pass query**:

  1.  Show the board WITHOUT the "Legal moves:" line.
  2.  Ask the model to output exactly one move coordinate.
  3.  Extract via regex `[a-h][1-8]` from raw output — NO fallback.
  4.  Check legality against `state.legal_moves()`.

This tests whether the model can:
  (a) Read and understand the ASCII board.
  (b) Identify a legal move from the board position.
  (c) Do so across multiple thinking budgets: [0, 64, 256, 512].

The resulting "legality-vs-B" curve directly measures OOD collapse.

─────────────────────────────────────────────────────────────────────────────
TWO QUERY MODES per budget
─────────────────────────────────────────────────────────────────────────────
  B=0 (no CoT):
    Single Ollama call, think=False, num_predict=32.
    Model must answer from pattern recognition alone.

  B>0 (with CoT):
    Pass-1: think=True, num_predict=B  →  captures reasoning trace.
    Pass-2: prepend reasoning, ask "Your move:", think=False, num_predict=32.
    Clear separation between thinking budget and answer budget.

─────────────────────────────────────────────────────────────────────────────
T=0.7, B=0 MEMORIZATION PROBE (optional, --no-t07 to skip)
─────────────────────────────────────────────────────────────────────────────
  Uncertain positions (UCT win-rate spread < threshold) are sampled K times at
  T=0.7 with no CoT. High consistency on uncertain positions → memorization.
  This probe still uses raw queries (no legal-moves list) for consistency.

Usage
─────
    # Quick OOD check — legality curve only:
    uv run python -u scripts/probe_ood.py --n-positions 15 --no-t07

    # Full probe including memorization:
    uv run python -u scripts/probe_ood.py --n-positions 30 --samples 10

    # Custom budgets:
    uv run python -u scripts/probe_ood.py --raw-budgets 0 64 128 256 512 --no-t07
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.games.reversi import ReversiState, initial_state


# ── System prompt for raw queries ─────────────────────────────────────────────

RAW_SYSTEM = (
    "You are playing Reversi/Othello on an 8×8 board.\n"
    "Pieces: B=Black, W=White, .=empty.\n"
    "Board columns are a–h (left→right). Rows are 1–8; row 8 is TOP, row 1 is BOTTOM.\n"
    "A coordinate like d3 means column d, row 3.\n"
    "A legal move must flip at least one opponent piece.\n"
    "Output ONLY a single move coordinate such as 'd3'. Nothing else."
)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class RawSample:
    """One raw-legality LLM query result."""
    position_id: int
    phase: str            # "early" / "mid" / "late"
    turn: int
    budget: int
    temperature: float
    raw_output: str       # full text from model (first 300 chars)
    extracted_move: str   # first [a-h][1-8] found, or "COLLAPSE"
    legal: bool
    uct_rank: int | None  # 1=best; None=outside top-3 or illegal
    pass1_tokens: int     # thinking tokens produced
    pass1_finish: str
    pass2_finish: str


@dataclass
class ProbePosition:
    """A board position annotated with UCT move evaluations."""
    position_id: int
    phase: str
    turn: int
    state: ReversiState
    uct_top3: list[dict]   # [{move, win_rate, visits}, ...]  sorted best-first
    uct_spread: float      # win_rate[0] - win_rate[2]  (inf if < 3 moves)
    uncertain: bool        # spread < threshold
    raw_samples: list[RawSample] = field(default_factory=list)
    memorization_samples: list[RawSample] = field(default_factory=list)


# ── Position generation ───────────────────────────────────────────────────────

def _phase_label(turn: int, total_turns: int) -> str:
    t = turn / max(total_turns, 1)
    if t < 0.33:
        return "early"
    if t < 0.67:
        return "mid"
    return "late"


async def _play_uct_game(rng: random.Random, iterations: int) -> list[ReversiState]:
    uct = UCTAgent(iterations=iterations)
    state = initial_state()
    states: list[ReversiState] = [state]
    while not state.is_terminal():
        telem = await uct.choose(state, seed=rng.randint(0, 2**31))
        state = state.apply_move(telem.chosen_move)
        states.append(state)
    return states


async def generate_probe_positions(
    n_positions: int,
    uct_game_iters: int,
    uct_eval_iters: int,
    spread_threshold: float,
    seed: int,
) -> list[ProbePosition]:
    rng = random.Random(seed)
    uct_eval = UCTAgent(iterations=uct_eval_iters)
    positions: list[ProbePosition] = []
    game_idx = 0

    while len(positions) < n_positions:
        game_idx += 1
        game_states = await _play_uct_game(rng, uct_game_iters)
        total = len(game_states)
        sampled_phases: set[str] = set()
        shuffled = list(enumerate(game_states))
        rng.shuffle(shuffled)

        for _, state in shuffled:
            if state.is_terminal() or len(state.legal_moves()) < 2:
                continue
            phase = _phase_label(state.turn_idx, total)
            if phase in sampled_phases:
                continue

            telem = await uct_eval.choose(state, seed=rng.randint(0, 2**31))
            top3 = telem.uct_top3

            if len(top3) >= 3:
                spread = top3[0]["win_rate"] - top3[2]["win_rate"]
            elif len(top3) == 2:
                spread = top3[0]["win_rate"] - top3[1]["win_rate"]
            else:
                spread = float("inf")

            positions.append(ProbePosition(
                position_id=len(positions),
                phase=phase,
                turn=state.turn_idx,
                state=state,
                uct_top3=top3,
                uct_spread=round(spread, 4),
                uncertain=(spread < spread_threshold),
            ))
            sampled_phases.add(phase)

            if len(positions) >= n_positions:
                break

    print(f"Generated {len(positions)} probe positions from {game_idx} games.")
    return positions


# ── Ollama HTTP helpers ───────────────────────────────────────────────────────

def _merge_response(data: dict) -> str:
    """Combine thinking + response fields (R1-Distill emits both)."""
    thinking = data.get("thinking") or ""
    response = data.get("response") or ""
    if thinking and response:
        return f"<think>{thinking}</think>\n{response}"
    return thinking or response or ""


def _extract_coordinate(text: str) -> str:
    """Return the model's chosen coordinate from raw output, or 'COLLAPSE'.

    Strategy:
    1. Strip <think>…</think> blocks to isolate the answer section.
    2. Return the LAST [a-h][1-8] match in the answer section — the model
       typically deliberates ("d3 is risky, e5 is better, f6 is best") and
       concludes with its final pick at the end.
    3. If the answer section is empty, fall back to the last coordinate in
       the full text (including thinking trace).
    """
    answer_part = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    matches = re.findall(r"\b([a-h][1-8])\b", answer_part, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    # Fallback: last coordinate anywhere in the full text.
    matches = re.findall(r"\b([a-h][1-8])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return "COLLAPSE"


async def query_raw(
    http: httpx.AsyncClient,
    pos: ProbePosition,
    budget: int,
    base_url: str,
    model: str,
    temperature: float = 0.0,
) -> RawSample:
    """Raw single/two-pass query WITHOUT a legal-moves list in the prompt.

    B=0  → single call, think=False, num_predict=32.
    B>0  → Pass-1 with think=True, num_predict=B; then Pass-2 with reasoning
            prepended, think=False, num_predict=32.
    """
    state = pos.state
    facing = state.current_player

    # Board rendered WITHOUT the "Legal moves:" line.
    board_text = state.render_text(facing=facing, show_legal=False)
    board_prompt = f"{board_text}\n\nYour move:"

    legal_set = {state.move_to_str(m) for m in state.legal_moves()}

    pass1_tokens = 0
    pass1_finish = "skipped(B=0)"

    if budget == 0:
        # ── Direct answer, no reasoning ──────────────────────────────────────
        body: dict = {
            "model": model,
            "system": RAW_SYSTEM,
            "prompt": board_prompt,
            "stream": False,
            "think": False,
            # 512 tokens so the model has room to reason through the
            # position and name a coordinate before being cut off.
            "options": {
                "num_predict": 512,
                "temperature": float(temperature),
            },
        }
        t0 = time.perf_counter()
        resp = await http.post(f"{base_url}/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
        # Use _merge_response: R1-Distill may put tokens in "thinking" even
        # with think=False; merging ensures we search the full output.
        raw_output = _merge_response(data).strip()
        pass2_finish = str(data.get("done_reason") or "stop")

    else:
        # ── Pass 1: free reasoning ────────────────────────────────────────────
        body1: dict = {
            "model": model,
            "system": RAW_SYSTEM,
            "prompt": board_prompt,
            "stream": False,
            "think": True,
            "options": {
                "num_predict": budget,
                "temperature": float(temperature),
            },
        }
        t0 = time.perf_counter()
        resp1 = await http.post(f"{base_url}/api/generate", json=body1)
        resp1.raise_for_status()
        data1 = resp1.json()
        reasoning = _merge_response(data1)
        pass1_tokens = int(data1.get("eval_count", 0))
        pass1_finish = str(data1.get("done_reason") or "stop")

        # ── Pass 2: extract answer from reasoning ─────────────────────────────
        pass2_prompt = (
            f"{board_prompt}\n\n"
            f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
            f"Your move (single coordinate only, e.g. d3):"
        )
        body2: dict = {
            "model": model,
            "prompt": pass2_prompt,
            "stream": False,
            "think": False,
            # 512 tokens — enough for the model to write a brief conclusion
            # and state its chosen coordinate explicitly.
            "options": {
                "num_predict": 512,
                "temperature": 0.0,
            },
        }
        resp2 = await http.post(f"{base_url}/api/generate", json=body2)
        resp2.raise_for_status()
        data2 = resp2.json()
        # Same: merge both fields so we don't miss the coordinate.
        raw_output = _merge_response(data2).strip()
        pass2_finish = str(data2.get("done_reason") or "stop")

    extracted = _extract_coordinate(raw_output)
    is_legal = extracted in legal_set

    uct_rank: int | None = None
    for rank, entry in enumerate(pos.uct_top3, 1):
        if entry["move"] == extracted:
            uct_rank = rank
            break

    return RawSample(
        position_id=pos.position_id,
        phase=pos.phase,
        turn=pos.turn,
        budget=budget,
        temperature=temperature,
        raw_output=raw_output[:300],
        extracted_move=extracted,
        legal=is_legal,
        uct_rank=uct_rank,
        pass1_tokens=pass1_tokens,
        pass1_finish=pass1_finish,
        pass2_finish=pass2_finish,
    )


# ── Summary printing ──────────────────────────────────────────────────────────

def _entropy(moves: list[str]) -> float:
    if not moves:
        return 0.0
    counts = Counter(moves)
    total = len(moves)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def print_raw_legality_results(positions: list[ProbePosition], budgets: list[int]) -> None:
    print("\n" + "=" * 66)
    print("RAW LEGALITY TEST — no legal-moves list, raw coordinate extraction")
    print("=" * 66)
    print(f"\n{'Budget':>8}  {'Legal':>8}  {'UCT-top1':>9}  {'UCT-top3':>9}  {'Collapse':>9}")
    print("-" * 66)

    for b in budgets:
        samples = [s for p in positions for s in p.raw_samples if s.budget == b]
        if not samples:
            continue
        n = len(samples)
        legal_n = sum(s.legal for s in samples)
        top1_n = sum(1 for s in samples if s.uct_rank == 1)
        top3_n = sum(1 for s in samples if s.uct_rank is not None)
        collapse_n = sum(1 for s in samples if s.extracted_move == "COLLAPSE")
        print(
            f"  B={b:<5}  {legal_n/n:>7.1%}  {top1_n/n:>8.1%}  {top3_n/n:>8.1%}  {collapse_n/n:>8.1%}"
            f"  ({legal_n}/{n})"
        )

    print("-" * 66)
    print()
    print("Interpretation:")
    print("  Legal%   — model produced a valid Reversi coordinate from the board alone.")
    print("  UCT-top1 — model's pick was the best move according to UCT-eval.")
    print("  UCT-top3 — model's pick was in the top-3 UCT moves.")
    print("  Collapse — model output contained no recognisable coordinate.")

    # OOD verdict
    b0_samples = [s for p in positions for s in p.raw_samples if s.budget == 0]
    if b0_samples:
        legal_frac = sum(s.legal for s in b0_samples) / len(b0_samples)
        collapse_frac = sum(1 for s in b0_samples if s.extracted_move == "COLLAPSE") / len(b0_samples)
        print()
        if collapse_frac > 0.5:
            print(
                f"  ⚠  OOD COLLAPSE CONFIRMED at B=0: {collapse_frac:.0%} of outputs contained no"
                " coordinate.\n"
                "     Report win-rate curves starting from B=64."
            )
        elif legal_frac < 0.5:
            print(
                f"  ⚠  WEAK OOD SIGNAL at B=0: {legal_frac:.0%} legal rate (below 50%)."
                " Model can produce coordinates but many are illegal.\n"
                "     Consider starting curves from B=64."
            )
        else:
            print(
                f"  ✓  No collapse at B=0: {legal_frac:.0%} of moves are legal."
                " Curves can start from B=0."
            )


def print_memorization_results(positions: list[ProbePosition]) -> None:
    uncertain = [p for p in positions if p.uncertain]
    if not uncertain:
        print("\n[Memorization probe] No uncertain positions — skipping.")
        return

    by_pos: dict[int, list[str]] = {}
    for p in uncertain:
        moves = [s.extracted_move for s in p.memorization_samples if s.extracted_move != "COLLAPSE"]
        if moves:
            by_pos[p.position_id] = moves

    if not by_pos:
        print("\n[Memorization probe] No valid samples.")
        return

    entropies = [_entropy(v) for v in by_pos.values()]
    mean_h = sum(entropies) / len(entropies)
    low_h = sum(1 for h in entropies if h < 0.5)

    print(f"\n[T=0.7, B=0 — memorization probe, {len(by_pos)} uncertain positions]")
    print(f"  Mean move entropy H: {mean_h:.3f} bits")
    print(f"  Low-entropy (<0.5 bits): {low_h}/{len(entropies)} positions")

    if mean_h < 0.5:
        print("  ⚠  Low entropy on genuinely uncertain positions → memorization signal.")
    elif mean_h > 1.5:
        print("  ✓  High entropy on uncertain positions → genuine uncertainty, not memorized.")
    else:
        print("  ~  Moderate entropy — inconclusive.")


# ── Output ────────────────────────────────────────────────────────────────────

def save_jsonl(positions: list[ProbePosition], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pos in positions:
            row: dict = {
                "position_id": pos.position_id,
                "phase": pos.phase,
                "turn": pos.turn,
                "uct_top3": pos.uct_top3,
                "uct_spread": pos.uct_spread,
                "uncertain": pos.uncertain,
                "raw_samples": [asdict(s) for s in pos.raw_samples],
                "memorization_samples": [asdict(s) for s in pos.memorization_samples],
            }
            f.write(json.dumps(row) + "\n")
    print(f"Results saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=180.0) as http:

        print(f"Generating {args.n_positions} probe positions …")
        positions = await generate_probe_positions(
            n_positions=args.n_positions,
            uct_game_iters=args.uct_game_iters,
            uct_eval_iters=args.uct_eval_iters,
            spread_threshold=args.spread_threshold,
            seed=args.seed,
        )
        n_uncertain = sum(1 for p in positions if p.uncertain)
        print(f"Uncertain positions (spread < {args.spread_threshold}): {n_uncertain}/{len(positions)}")

        # ── Raw legality sweep ────────────────────────────────────────────────
        total_queries = len(positions) * len(args.raw_budgets)
        print(
            f"\n── Raw legality sweep: {len(positions)} positions × "
            f"{len(args.raw_budgets)} budgets {args.raw_budgets} = {total_queries} queries ──"
        )
        q = 0
        for pos in positions:
            for budget in args.raw_budgets:
                q += 1
                sample = await query_raw(
                    http, pos, budget,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=0.0,
                )
                pos.raw_samples.append(sample)
                status = "✓ legal" if sample.legal else f"✗ '{sample.extracted_move}'"
                rank_str = f"UCT-rank={sample.uct_rank}" if sample.uct_rank else "outside top-3"
                print(
                    f"  [{q:3d}/{total_queries}] B={budget:<5} "
                    f"pos={pos.position_id} phase={pos.phase} turn={pos.turn:2d}  "
                    f"{status}  {rank_str}"
                )

        print_raw_legality_results(positions, args.raw_budgets)

        # ── T=0.7, B=0 memorization probe ────────────────────────────────────
        if not args.no_t07:
            uncertain_pos = [p for p in positions if p.uncertain]
            if not uncertain_pos:
                print("\nNo uncertain positions — skipping memorization probe.")
            else:
                n_q = len(uncertain_pos) * args.samples
                print(
                    f"\n── T=0.7, B=0 memorization probe: "
                    f"{len(uncertain_pos)} positions × {args.samples} samples = {n_q} queries ──"
                )
                q = 0
                for pos in uncertain_pos:
                    print(
                        f"  pos={pos.position_id} phase={pos.phase} "
                        f"turn={pos.turn} spread={pos.uct_spread:.3f}",
                        end="  ", flush=True,
                    )
                    chosen: list[str] = []
                    for _ in range(args.samples):
                        q += 1
                        sample = await query_raw(
                            http, pos, budget=0,
                            base_url=args.base_url,
                            model=args.model,
                            temperature=0.7,
                        )
                        pos.memorization_samples.append(sample)
                        chosen.append(sample.extracted_move)
                    h = _entropy([m for m in chosen if m != "COLLAPSE"])
                    top = Counter(chosen).most_common(3)
                    print(f"H={h:.2f} bits  {top}")

            print_memorization_results(positions)

        out_path = Path(args.output)
        save_jsonl(positions, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe R1-Distill OOD collapse (raw legality, no legal-moves list).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n-positions", type=int, default=15,
                    help="Probe positions to generate.")
    ap.add_argument("--raw-budgets", type=int, nargs="+", default=[0, 64, 256, 512],
                    help="Thinking-token budgets to test (space-separated).")
    ap.add_argument("--samples", type=int, default=10,
                    help="Samples per uncertain position for T=0.7 memorization probe.")
    ap.add_argument("--uct-game-iters", type=int, default=50,
                    help="UCT iterations for self-play position generation.")
    ap.add_argument("--uct-eval-iters", type=int, default=100,
                    help="UCT iterations for per-position move quality evaluation.")
    ap.add_argument("--spread-threshold", type=float, default=0.10,
                    help="Win-rate spread threshold for 'uncertain' positions.")
    ap.add_argument("--no-t07", action="store_true",
                    help="Skip T=0.7 memorization probe.")
    ap.add_argument("--model", default="deepseek-r1:7b",
                    help="Ollama model tag.")
    ap.add_argument("--base-url", default="http://localhost:11434",
                    help="Ollama base URL.")
    ap.add_argument("--output", default="data/probes/ood_probe.jsonl",
                    help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = ap.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
