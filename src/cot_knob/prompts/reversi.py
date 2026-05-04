"""Reversi prompt templates — board-centric design.

The system message contains rules only (no strategy advice).
The user turn for Pass 1 shows:
  - The ASCII board (rows 8→1, B/W notation), which is compact and directly
    readable — the model no longer needs to re-derive grid coordinates from
    a nested JSON array.
  - The move history rendered by the memory manager.

Variants:

- ``free_cot``       — "Reason about your move." (proposal default)
- ``structured_cot`` — explicit three-step scaffold (Task 8 ablation)
"""

from __future__ import annotations

from typing import Literal

from cot_knob.games.reversi import ReversiState

PromptVariant = Literal["free_cot", "structured_cot"]

# ── System prompts ──────────────────────────────────────────────────────────

# Pass 1 (free reasoning).  Rules only — no hints, no opening theory.
SYSTEM_PROMPT = """You are playing Reversi (Othello) on an 8×8 board.

Rules:
- Black (B) moves first; players alternate turns.
- To place a piece you must outflank at least one opponent piece — trapping it between your new piece and another of yours along a row, column, or diagonal. All trapped opponent pieces flip to your colour.
- You must pass if you have no legal move.
- The game ends when neither player can move. The player with more pieces wins.

CRITICAL — begin your reasoning IMMEDIATELY with "Legal moves: ..." listing the candidate moves from the board footer. Do NOT open with "Okay", "Sure", "Let me", "I need to", "I'm trying to figure out", or any other preamble. Do NOT re-read or re-describe the board — it is correct as given. Do NOT recite these rules back."""

# Pass 2 (constrained choice).  Kept minimal — one token answer only.
SYSTEM_PROMPT_PASS2 = """Same Reversi rules apply.
Output exactly ONE move token from the list you are given. No explanation, no punctuation, just the move notation (e.g. c4) or the word pass."""

# ── User templates ──────────────────────────────────────────────────────────

REASON_TEMPLATE_FREE = """{board}

Move history (oldest → newest):
{memory_text}

Reason about your next move inside <reasoning>...</reasoning>.
Do NOT reprint the board or the move list inside <reasoning>.
After </reasoning>, write on its own line:  ANSWER: <move>"""

REASON_TEMPLATE_STRUCTURED = """{board}

Move history (oldest → newest):
{memory_text}

Inside <reasoning>...</reasoning> follow ONLY these three steps:
1. Name the legal moves you will compare (copy from the Legal line above).
2. One pro and one con for each candidate.
3. State your chosen move and a one-sentence justification.
Do NOT reprint the board inside <reasoning>.
After </reasoning>, write on its own line:  ANSWER: <move>"""

SELECT_TEMPLATE = """{prior}

Choose exactly one of: {choices}"""


# ── Render helpers ───────────────────────────────────────────────────────────

def render_reason_prompt(
    state: ReversiState,
    *,
    memory_text: str,
    variant: PromptVariant = "free_cot",
    facing: int = 1,
) -> str:
    """Build the Pass-1 (reasoning) user prompt.

    ``facing`` (+1 or -1) controls the "You play B/W" label in the board footer
    and should match the side the LLM agent is playing.
    """
    board = state.render_text(facing=facing)
    template = REASON_TEMPLATE_STRUCTURED if variant == "structured_cot" else REASON_TEMPLATE_FREE
    return template.format(board=board, memory_text=memory_text)


def render_select_prompt(
    state: ReversiState,
    *,
    reason_prompt: str,
    pass1_text: str,
) -> tuple[str, list[str]]:
    """Build the Pass-2 (constrained selection) prompt + choices list."""
    legal = state.legal_moves()
    choices = [state.move_to_str(m) for m in legal] or ["pass"]
    prior = (
        f"{reason_prompt}\n\n"
        f"<reasoning>\n{pass1_text}\n</reasoning>"
    )
    return SELECT_TEMPLATE.format(prior=prior, choices=", ".join(choices)), choices
