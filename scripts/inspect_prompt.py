"""Inspect exactly what goes into the model before reasoning — no LLM calls.

Simulates N turns of a Nim game and prints the complete system prompt +
user prompt for each LLM turn, exactly as the agent would send them.
Useful for auditing the memory system and prompt construction.

Usage::

    # Walk through 3 LLM turns of a sample game, step_by_step variant
    uv run python scripts/inspect_prompt.py --variant step_by_step --turns 3

    # Show what the model sees on LLM turn 2 with free_cot
    uv run python scripts/inspect_prompt.py --variant free_cot --llm-turn 2

    # Custom pile sequence
    uv run python scripts/inspect_prompt.py --piles 7 11 13 --variant step_by_step

Options
-------
--piles       Pile sizes (default: 3 5 7)
--variant     Prompt variant: free_cot | step_by_step | nim_sum_given | few_shot
--turns       Number of LLM turns to walk through (default: 3)
--llm-turn    Show only this specific LLM turn index (0-based); overrides --turns
--seed        RNG seed for opponent moves (default: 0)
--llm-side    1 or -1 (default: 1, LLM plays first)
--budget      Budget B to annotate on each turn (default: 512)
--no-color    Plain text output without ANSI colours
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cot_knob.games.nim import NimState
from cot_knob.memory.base import TurnRecord
from cot_knob.memory.last_move import LastMoveMemory
import cot_knob.prompts.nim as nim_prompts


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(code: str, text: str, use_color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if use_color else text

def _header(title: str, use_color: bool) -> str:
    bar = "═" * 70
    return _c("1;36", f"\n{bar}\n  {title}\n{bar}", use_color)

def _section(label: str, use_color: bool) -> str:
    return _c("1;33", f"\n── {label} {'─'*(66-len(label))}", use_color)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--piles", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--variant", default="step_by_step",
                   choices=["free_cot", "step_by_step", "nim_sum_given", "few_shot"])
    p.add_argument("--turns", type=int, default=3,
                   help="Number of LLM turns to show (default: 3)")
    p.add_argument("--llm-turn", type=int, default=None,
                   help="Show only this LLM turn (0-based). Overrides --turns.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm-side", type=int, default=1, choices=[1, -1])
    p.add_argument("--budget", type=int, default=512,
                   help="Budget B shown in header (doesn't affect prompt content)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    rng = random.Random(args.seed)
    state = NimState(tuple(args.piles))
    memory = LastMoveMemory()

    system_prompt = nim_prompts.get_system_prompt(args.variant)

    show_only = args.llm_turn  # None → show first --turns LLM turns
    max_llm_turns = 1 if show_only is not None else args.turns

    llm_turn_count = 0
    game_turn = 0

    print(_header(
        f"Nim prompt inspector — piles={args.piles}  variant={args.variant}  "
        f"B={args.budget}  llm_side={'P1' if args.llm_side == 1 else 'P2'}",
        use_color,
    ))

    while not state.is_terminal() and llm_turn_count < max_llm_turns:
        current_player = state.current_player
        is_llm = current_player == args.llm_side

        if is_llm:
            if show_only is None or llm_turn_count == show_only:
                # ── Print exactly what goes into the model ──────────────────
                mem_snap = await memory.render()
                user_prompt = nim_prompts.render_reason_prompt(
                    state,
                    memory_text=mem_snap.text,
                    variant=args.variant,
                    facing=args.llm_side,
                )

                print(_header(
                    f"LLM TURN {llm_turn_count}  (game turn {game_turn})  "
                    f"player={'P1' if current_player==1 else 'P2'}  B={args.budget}",
                    use_color,
                ))

                print(_section("SYSTEM PROMPT", use_color))
                print(system_prompt)

                print(_section("USER PROMPT (full content sent to model)", use_color))
                print(user_prompt)

                # ── Annotate what memory contributed ──────────────────────
                print(_section("MEMORY TEXT (what the memory system injected)", use_color))
                print(_c("32", repr(mem_snap.text), use_color))

                # ── Board at this moment ───────────────────────────────────
                print(_section("CURRENT BOARD (from live game state)", use_color))
                print(state.render_text(show_legal=True))
                print(f"  Piles (raw): {list(state.piles)}")
                print(f"  Nim-sum: {state.piles[0]}", end="")
                ns = 0
                for px in state.piles:
                    ns ^= px
                    print(f" XOR {px}", end="") if px != state.piles[0] else None
                print(f" = {ns}  ({'winning' if ns != 0 else 'losing'} position)")

                print()

            llm_turn_count += 1
            if llm_turn_count >= max_llm_turns and show_only is None:
                break

            # Make a random move for the LLM (we just want to advance state)
            legal = state.legal_moves()
            move_id = rng.choice(legal)

        else:
            # Opponent: random move
            legal = state.legal_moves()
            move_id = rng.choice(legal)

        move_str = state.move_to_str(move_id)

        if is_llm or True:  # always update memory so it's accurate
            # Advance state FIRST (matching the fixed runner.py order)
            state_before = state
            state = state.apply_move(move_id)
            rec = TurnRecord(
                turn_idx=game_turn,
                player=current_player,
                move_str=move_str,
                state_after_serialized=state.to_serializable(),  # post-move ← correct
            )
            memory.update(rec)
            if not is_llm:
                agent_label = _c("35", f"OPP (P{'1' if current_player==1 else '2'})", use_color)
                print(f"  [game turn {game_turn}] {agent_label} plays: {move_str}  "
                      f"→ piles now {list(state.piles)}")
        else:
            state = state.apply_move(move_id)

        game_turn += 1

    if state.is_terminal():
        print(_c("1;32", "\n  [Game over — terminal state reached during inspection]", use_color))


if __name__ == "__main__":
    asyncio.run(main())
