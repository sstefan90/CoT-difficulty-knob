"""LLM agent — two-pass architecture for both Reversi and Nim.

Both games use the same two-pass design:
  Pass 1: free reasoning, max_tokens=B.  May be truncated at low B — that is the
          experiment variable.  No move is extracted here.
  Pass 2: constrained selection from the enumerated legal moves via generate_choice.
          Always produces a valid move regardless of Pass-1 quality or truncation.

This cleanly separates reasoning quality (B-dependent) from move legality
(structurally guaranteed), eliminating the mid/late-game truncation confound:
as memory_text grows across turns the model's attention is pulled toward history,
but Pass-2 still picks a valid move even if Pass-1 reasoning was cut short.

Legacy single-pass artefacts (MOVE tag parsing, Tier-2/3 fallbacks) are removed.
"""

from __future__ import annotations

import random
from typing import Literal

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState
from cot_knob.llm.client import LLMClient
from cot_knob.memory.base import MemoryManager
from cot_knob.prompts.reversi import PromptVariant


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
        """Nim two-pass path — mirrors the Reversi architecture.

        Pass 1: up to B tokens of free reasoning.  May be truncated at low B —
                that is the experiment's independent variable.  No move is read
                from this text; truncation is NOT a failure mode.

        Pass 2: generate_choice() over the legal move strings, with the Pass-1
                reasoning in context.  Always produces a valid move; parse_failed
                is structurally impossible.

        This eliminates the mid/late-game truncation confound: as memory_text
        grows across turns the probability that Pass-1 reasoning is cut short
        increases, but Pass-2 still picks a valid move regardless.
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
        system_prompt_p2 = nim_prompts.SYSTEM_PROMPT_PASS2

        # ── Pass 1: free reasoning ────────────────────────────────────────────
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

        # ── Pass 2: constrained move selection ────────────────────────────────
        # Build the selection prompt from the Pass-1 reasoning, then pick one
        # legal move via generate_choice (constrained — always valid).
        select_prompt, choices = nim_prompts.render_select_prompt(
            state,  # type: ignore[arg-type]
            reason_prompt=reason_prompt,
            pass1_text=pass1_text,
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
                "pass2_prompt_tokens": choice.n_input_tokens,
                "pass2_cached_tokens": (
                    (choice.raw.get("usage") or {})
                    .get("prompt_tokens_details") or {}
                ).get("cached_tokens"),
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
                "pass2_prompt_tokens": choice.n_input_tokens,
                "pass2_cached_tokens": (
                    (choice.raw.get("usage") or {})
                    .get("prompt_tokens_details") or {}
                ).get("cached_tokens"),
            },
        )
