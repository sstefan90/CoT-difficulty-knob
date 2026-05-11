"""Nim prompt templates.

Mirrors the structure of ``cot_knob.prompts.reversi`` so the same LLMAgent
design works for both games with a single ``game`` parameter switch.

Move output format: "MOVE: pile=A take=3"  (structured tag, no legal enumeration).
Legal moves are NOT listed in the prompt — the model is expected to derive them
from the pile sizes shown on the board. This is valid because the OOD legality
probe confirmed 100% legality at all budgets including B=0.

Prompt variants
---------------
free_cot        — baseline: board + brief instruction (default)
nim_sum_given   — diagnostic (a): board + pre-computed XOR nim-sum injected
step_by_step    — diagnostic (b): structured scaffold walking through XOR steps
few_shot        — diagnostic (c): worked example of nim-sum in the system prompt
"""

from __future__ import annotations

from functools import reduce
from operator import xor

from cot_knob.games.nim import NimState

# ── System prompts ────────────────────────────────────────────────────────────

# Baseline — rules only, no strategy hints.
SYSTEM_PROMPT = """You are playing Nim.

Rules:
- The board shows piles of stones labelled A, B, C, ...
- On your turn remove at least 1 stone from exactly one non-empty pile. You may remove any number up to the full pile.
- The player who takes the very last stone wins (normal play convention).

After your reasoning, end your response with EXACTLY this line (nothing after it):
MOVE: pile=X take=N
where X is the pile letter (A, B, C, ...) and N is the number of stones to take."""

# Diagnostic (b) — step-by-step XOR scaffold.
SYSTEM_PROMPT_STEP_BY_STEP = """You are playing Nim.

Rules:
- The board shows piles of stones labelled A, B, C, ...
- On your turn remove at least 1 stone from exactly one non-empty pile.
- The player who takes the very last stone wins (normal play convention).

Winning strategy — nim-sum (XOR):
Work through these steps in order:
1. Write each pile size in binary.
2. XOR the bits column by column to get the nim-sum.
3. If nim-sum = 0 you are in a losing position; make any legal move.
4. If nim-sum ≠ 0, find the pile P where: (P XOR nim-sum) < P.
   Take enough stones from P so the new pile size equals (P XOR nim-sum).
   Verify by recomputing the XOR — it must equal 0 after your move.

After completing the steps, end your response with EXACTLY:
MOVE: pile=X take=N"""

# Diagnostic (c) — few-shot worked example in the system prompt.
SYSTEM_PROMPT_FEW_SHOT = """You are playing Nim.

Rules:
- The board shows piles of stones labelled A, B, C, ...
- On your turn remove at least 1 stone from exactly one non-empty pile.
- The player who takes the very last stone wins (normal play convention).

How to find the winning move (nim-sum / XOR method):

WORKED EXAMPLE — Piles: A=3, B=5, C=7
  Binary:   A = 011
            B = 101
            C = 111
  XOR:        011
            ^ 101
            ^ 111
            -----
              001   ← nim-sum = 1  (non-zero → winning position)

  Goal: make nim-sum = 0 after your move.
  Try pile A (= 3 = 011): new A needed = 011 XOR 001 = 010 = 2.
  Take 1 from A → piles become A=2, B=5, C=7.
  Verify: 010 XOR 101 XOR 111 = 000 ✓

  MOVE: pile=A take=1

Apply the same procedure to the current position, then end with:
MOVE: pile=X take=N"""

# Pass 2 (constrained choice) — used by both Nim and Reversi paths.
# The model has already reasoned in Pass 1; here it simply commits to one move.
SYSTEM_PROMPT_PASS2 = """Same Nim rules apply.
Output exactly ONE move from the list you are given. No explanation, no punctuation — just the move string (e.g. take 2 from A)."""

# ── User templates ────────────────────────────────────────────────────────────

REASON_TEMPLATE_FREE = """{board}

{memory_text}

Reason about the pile sizes and nim-sum strategy. End your response with exactly:
MOVE: pile=X take=N"""

# Diagnostic (a) — XOR nim-sum pre-computed and injected into the user message.
REASON_TEMPLATE_NIM_SUM_GIVEN = """{board}

Current XOR nim-sum of the piles: {nim_sum}
(If nim-sum = 0 you are in a losing position; if nim-sum ≠ 0 you can force a win.)

{memory_text}

Using the nim-sum above, determine your move and end with exactly:
MOVE: pile=X take=N"""

SELECT_TEMPLATE = """{prior}

Choose exactly one of: {choices}"""


# ── Render helpers ────────────────────────────────────────────────────────────

def _nim_sum(piles: tuple[int, ...]) -> int:
    return reduce(xor, piles, 0)


def render_reason_prompt(
    state: NimState,
    *,
    memory_text: str,
    variant: str = "free_cot",
    facing: int = 1,
) -> str:
    """Build the Pass-1 (reasoning) user prompt for Nim.

    Legal moves are NOT enumerated — the model derives them from pile sizes.
    ``facing`` is accepted for API compatibility with Reversi but unused.

    Variants:
      free_cot        — board only, no hints (default)
      nim_sum_given   — board + pre-computed XOR nim-sum
      step_by_step    — same as free_cot (system prompt carries the scaffold)
      few_shot        — same as free_cot (system prompt carries the example)
    """
    board = state.render_text(show_legal=False)

    if variant == "nim_sum_given":
        ns = _nim_sum(state.piles)
        return REASON_TEMPLATE_NIM_SUM_GIVEN.format(
            board=board, memory_text=memory_text, nim_sum=ns
        )

    # step_by_step and few_shot: the system prompt carries all extra structure;
    # the user message is the same as free_cot.
    return REASON_TEMPLATE_FREE.format(board=board, memory_text=memory_text)


def get_system_prompt(variant: str = "free_cot") -> str:
    """Return the correct system prompt for a given variant."""
    if variant == "step_by_step":
        return SYSTEM_PROMPT_STEP_BY_STEP
    if variant == "few_shot":
        return SYSTEM_PROMPT_FEW_SHOT
    return SYSTEM_PROMPT  # free_cot and nim_sum_given use the baseline


def render_select_prompt(
    state: NimState,
    *,
    reason_prompt: str,
    pass1_text: str,
) -> tuple[str, list[str]]:
    """Build the Pass-2 (constrained selection) prompt + choices list.

    Called for both Nim and Reversi. The Pass-1 reasoning is included as
    context so the model's commit reflects its own analysis.
    """
    legal = state.legal_moves()
    choices = [state.move_to_str(m) for m in legal]
    if not choices:
        choices = ["pass"]
    prior = (
        f"{reason_prompt}\n\n"
        f"<reasoning>\n{pass1_text}\n</reasoning>"
    )
    return SELECT_TEMPLATE.format(prior=prior, choices=", ".join(choices)), choices
