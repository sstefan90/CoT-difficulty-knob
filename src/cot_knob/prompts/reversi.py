"""Reversi prompt templates.

Mirrors the proposal's "CoT prompt structure" section verbatim where
possible, parameterized only by budget B and the prompt variant.

Variants:

- ``free_cot``      — "Think through your move." (proposal default)
- ``structured_cot`` — explicit four-step scaffolding (Task 8 ablation)
"""

from __future__ import annotations

import json
from typing import Literal

from cot_knob.games.reversi import ReversiState

PromptVariant = Literal["free_cot", "structured_cot"]

SYSTEM_PROMPT = """You are an expert Reversi (Othello) player on the 8x8 board.

Rules:
- Black (X) moves first. Players alternate.
- A legal move places one of your pieces on an empty square such that it flanks
  one or more of the opponent's pieces between the new piece and another of your
  pieces along a row, column, or diagonal. All flanked opponent pieces flip.
- If you have no legal moves you must pass.
- Game ends when neither player can move; whoever has more pieces wins.

Strategic considerations:
- Corners are stable and very valuable.
- X-squares (b2, b7, g2, g7) and C-squares (a2/b1 etc.) often surrender corners; avoid them
  unless forced.
- Maximize your *frontier discs* (pieces adjacent to empty squares) for the *opponent*,
  not yourself.
- Mobility (number of legal moves) tends to matter more than disc count in the early game.

Format requirements:
- Respond with your reasoning inside <reasoning>...</reasoning> tags.
- Then on a new line, write exactly: ANSWER: <move>  (e.g. ANSWER: c4 or ANSWER: pass)
- The move must be one of the legal moves listed in the prompt.
"""

REASON_TEMPLATE_FREE = """Current game state:
{state_json}

Board:
{board_text}

Legal moves:
{legal_moves_enum}

Recent history / memory:
{memory_text}

Think through your move."""

REASON_TEMPLATE_STRUCTURED = """Current game state:
{state_json}

Board:
{board_text}

Legal moves:
{legal_moves_enum}

Recent history / memory:
{memory_text}

Think through your move using exactly these steps:
1. Identify any threats from the opponent (corners, edge swings, parity issues).
2. List the 2-3 strongest candidate moves from the legal moves above.
3. Evaluate each candidate (corner risk, mobility delta, frontier control).
4. State which move you will play and why."""

SELECT_TEMPLATE = """{prior}

Based on the reasoning above, output exactly one of the legal moves: {choices}."""


def render_reason_prompt(
    state: ReversiState,
    *,
    memory_text: str,
    variant: PromptVariant = "free_cot",
) -> str:
    legal = state.legal_moves()
    legal_enum = ", ".join(f"{i}: {state.move_to_str(m)}" for i, m in enumerate(legal))
    state_json = json.dumps(state.to_serializable(), separators=(",", ":"))
    template = REASON_TEMPLATE_STRUCTURED if variant == "structured_cot" else REASON_TEMPLATE_FREE
    return template.format(
        state_json=state_json,
        board_text=state.render_text(),
        legal_moves_enum=legal_enum or "(none — must pass)",
        memory_text=memory_text,
    )


def render_select_prompt(
    state: ReversiState,
    *,
    reason_prompt: str,
    pass1_text: str,
) -> tuple[str, list[str]]:
    legal = state.legal_moves()
    choices = [state.move_to_str(m) for m in legal] or ["pass"]
    prior = (
        f"{reason_prompt}\n\n"
        f"<reasoning>\n{pass1_text}\n</reasoning>"
    )
    return SELECT_TEMPLATE.format(prior=prior, choices=", ".join(choices)), choices
