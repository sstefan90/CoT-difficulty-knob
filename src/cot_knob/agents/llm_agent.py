"""LLM agent — two-pass (Reversi) or single-pass with MOVE tag (Nim).

Reversi path:
  Pass 1: free reasoning, max_tokens=B.
  Pass 2: constrained selection over the enumerated legal moves via generate_choice.

Nim path (single-pass, no legal-move enumeration):
  Pass 1: B tokens of reasoning. Model instructed to end with "MOVE: pile=X take=N".
  No Pass 2 — move extracted from Pass-1 text via two-tier regex.

  For Llama (think=False): response field is the complete reasoning + MOVE tag.
  For R1 (think=True): response field only appears if thinking finishes; at low
  budgets reasoning is truncated and there may be no MOVE tag (parse_failed).

All telemetry for SQLite ``turns`` and ``model_calls`` rows is on TurnTelemetry.
"""

from __future__ import annotations

import random
import re
from typing import Literal

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState
from cot_knob.llm.client import LLMClient
from cot_knob.memory.base import MemoryManager
from cot_knob.prompts.reversi import PromptVariant

# MOVE tag: the structured final line the model is prompted to write.
# Search for the LAST occurrence so the model's final revision wins.
_NIM_MOVE_TAG_RE = re.compile(
    r"MOVE:\s*pile=([A-Za-z])\s+take=(\d+)", re.IGNORECASE
)

# Fallback patterns — catch "take N from X", "N stones from X" etc.
# Used when the formal MOVE tag is absent (e.g. R1 with truncated thinking).
_NIM_TAKE_REGEXES: list[re.Pattern[str]] = [
    re.compile(
        r"(?:take|taking|remove|removing)\s+(?:away\s+)?(\d+)\s+(?:stone[s]?\s+)?from\s+(?:pile\s+)?([A-Za-z])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+)\s+stone[s]?\s+from\s+(?:pile\s+)?([A-Za-z])\b",
        re.IGNORECASE,
    ),
]


def _extract_nim_move_tag(text: str) -> tuple[str, int] | None:
    """Return (pile_letter_upper, stones) from the last MOVE tag, or None."""
    matches = _NIM_MOVE_TAG_RE.findall(text)
    if not matches:
        return None
    pile_letter, stones_str = matches[-1]
    return pile_letter.upper(), int(stones_str)


class LLMAgent(Agent):
    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        memory: MemoryManager,
        *,
        budget: int,
        game: Literal["reversi", "nim"] = "reversi",
        prompt_variant: PromptVariant = "free_cot",
        temperature: float = 0.0,
        side: Literal[1, -1] = 1,
    ) -> None:
        self._client = client
        self._memory = memory
        self.budget = int(budget)
        self.game = game
        self.prompt_variant: PromptVariant = prompt_variant
        self.temperature = float(temperature)
        self.side = side

    def _get_prompts(self):
        """Return (SYSTEM_PROMPT, SYSTEM_PROMPT_PASS2, render_reason, render_select)."""
        if self.game == "nim":
            import cot_knob.prompts.nim as nim_prompts
            return (
                nim_prompts.SYSTEM_PROMPT,
                nim_prompts.SYSTEM_PROMPT_PASS2,
                nim_prompts.render_reason_prompt,
                nim_prompts.render_select_prompt,
            )
        import cot_knob.prompts.reversi as rev_prompts
        return (
            rev_prompts.SYSTEM_PROMPT,
            rev_prompts.SYSTEM_PROMPT_PASS2,
            rev_prompts.render_reason_prompt,
            rev_prompts.render_select_prompt,
        )

    async def _choose_nim(
        self,
        state: GameState,
        legal: list[int],
        *,
        seed: int | None,
    ) -> TurnTelemetry:
        """Nim single-pass path.

        Calls generate() with no explicit think override — the client's instance
        default (set from model config) applies.  For Llama (think=False) this
        means the full response text is returned in one field and reliably ends
        with the MOVE tag.  For R1 (think=True) the MOVE tag may be absent at
        low budgets (truncated thinking) and parse_failed fires.

        Extraction tiers:
          1. Formal MOVE tag (last occurrence).
          2. Last "take N from X" / "N stones from X" mention (fallback for R1).
          3. Random legal move + parse_failed=True.
        """
        import cot_knob.prompts.nim as nim_prompts

        mem = await self._memory.render()
        reason_prompt = nim_prompts.render_reason_prompt(
            state,  # type: ignore[arg-type]
            memory_text=mem.text,
            variant=self.prompt_variant,
            facing=self.side,
        )
        system_prompt = nim_prompts.get_system_prompt(self.prompt_variant)

        if self.budget > 0:
            comp = await self._client.generate(
                reason_prompt,
                max_tokens=self.budget,
                temperature=self.temperature,
                seed=seed,
                system=system_prompt,
                # think=None → uses client default (False for Llama, True for R1)
            )
            pass1_text = comp.text
            pass1_tokens_in = comp.n_input_tokens
            pass1_tokens_out = comp.n_output_tokens
            pass1_finish = comp.finish_reason
            pass1_latency = comp.latency_ms
        else:
            pass1_text = ""
            pass1_tokens_in = 0
            pass1_tokens_out = 0
            pass1_finish = "skipped"
            pass1_latency = 0.0

        # ── Tier 1: formal MOVE tag ───────────────────────────────────────────
        chosen_move: int | None = None
        chosen_str = ""

        tag = _extract_nim_move_tag(pass1_text)
        if tag is not None:
            pile_letter, stones = tag
            chosen_str = f"take {stones} from {pile_letter}"
            chosen_move = state.str_to_move(chosen_str)

        # ── Tier 2: last take-like mention (R1 truncated reasoning fallback) ──
        if chosen_move is None or chosen_move not in legal:
            for take_re in _NIM_TAKE_REGEXES:
                take_mentions = take_re.findall(pass1_text)
                for stones_str, pile_letter in reversed(take_mentions):
                    attempt = f"take {stones_str} from {pile_letter.upper()}"
                    candidate = state.str_to_move(attempt)
                    if candidate is not None and candidate in legal:
                        chosen_move = candidate
                        chosen_str = attempt
                        break
                if chosen_move is not None and chosen_move in legal:
                    break

        # ── Tier 3: random legal fallback ─────────────────────────────────────
        parse_failed = False
        if chosen_move is None or chosen_move not in legal:
            rng = random.Random(seed)
            chosen_move = rng.choice(legal)
            chosen_str = state.move_to_str(chosen_move)
            parse_failed = True

        return TurnTelemetry(
            chosen_move=chosen_move,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            llm_pass1_prompt=reason_prompt,
            llm_pass1_system=system_prompt,
            llm_pass1_text=pass1_text,
            llm_pass1_tokens_in=pass1_tokens_in,
            llm_pass1_tokens_out=pass1_tokens_out,
            llm_pass1_finish=pass1_finish,
            llm_pass1_latency_ms=pass1_latency,
            llm_pass2_choice_text=chosen_str,
            llm_pass2_tokens_out=0,
            llm_pass2_finish="skipped",
            llm_pass2_latency_ms=0.0,
            llm_pass2_parse_failed=parse_failed,
            extra={
                "agent": "llm",
                "budget": self.budget,
                "prompt_variant": self.prompt_variant,
                "memory_kind": mem.kind,
                "memory_n_tokens_est": mem.n_tokens_estimate,
            },
        )

    async def choose(self, state: GameState, *, seed: int | None = None) -> TurnTelemetry:

        legal = state.legal_moves()
        if not legal:
            return TurnTelemetry(
                chosen_move=-1,
                legal_moves=[],
                state_serialized=state.to_serializable(),
                extra={"agent": "llm", "passed": True, "budget": self.budget},
            )

        if self.game == "nim":
            return await self._choose_nim(state, legal, seed=seed)

        # ── Reversi two-pass path ──────────────────────────────────────────────
        system_prompt, system_prompt_p2, render_reason, render_select = self._get_prompts()

        mem = await self._memory.render()
        reason_prompt = render_reason(
            state, memory_text=mem.text, variant=self.prompt_variant, facing=self.side
        )
        if self.budget > 0:
            comp = await self._client.generate(
                reason_prompt,
                max_tokens=self.budget,
                temperature=self.temperature,
                seed=seed,
                system=system_prompt,
                # think=None → uses client default
            )
            pass1_text = comp.text
            pass1_tokens_in = comp.n_input_tokens
            pass1_tokens_out = comp.n_output_tokens
            pass1_finish = comp.finish_reason
            pass1_latency = comp.latency_ms
        else:
            pass1_text = ""
            pass1_tokens_in = 0
            pass1_tokens_out = 0
            pass1_finish = "skipped"
            pass1_latency = 0.0

        select_prompt, choices = render_select(
            state, reason_prompt=reason_prompt, pass1_text=pass1_text
        )
        choice = await self._client.generate_choice(
            select_prompt,
            choices=choices,
            temperature=self.temperature,
            seed=seed,
            system=system_prompt_p2,
        )
        parse_failed = bool(choice.raw.get("__choice_parse_failed", False))

        chosen_str = choice.choice_text
        chosen_move = state.str_to_move(chosen_str)
        if chosen_move is None or chosen_move not in legal:
            rng = random.Random(seed)
            chosen_move = rng.choice(legal)
            parse_failed = True

        return TurnTelemetry(
            chosen_move=chosen_move,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            llm_pass1_prompt=reason_prompt,
            llm_pass1_system=system_prompt,
            llm_pass1_text=pass1_text,
            llm_pass1_tokens_in=pass1_tokens_in,
            llm_pass1_tokens_out=pass1_tokens_out,
            llm_pass1_finish=pass1_finish,
            llm_pass1_latency_ms=pass1_latency,
            llm_pass2_choice_text=chosen_str,
            llm_pass2_tokens_out=choice.n_output_tokens,
            llm_pass2_finish=choice.finish_reason,
            llm_pass2_latency_ms=choice.latency_ms,
            llm_pass2_parse_failed=parse_failed,
            extra={
                "agent": "llm",
                "budget": self.budget,
                "prompt_variant": self.prompt_variant,
                "memory_kind": mem.kind,
                "memory_n_tokens_est": mem.n_tokens_estimate,
            },
        )
