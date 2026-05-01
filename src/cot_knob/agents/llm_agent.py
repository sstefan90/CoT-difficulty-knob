"""Two-pass LLM agent.

Implements the proposal's primary agent design:

  Pass 1: free reasoning, ``max_tokens = B``  (the budget knob).
  Pass 2: constrained move selection over the enumerated legal moves.

All telemetry needed for the SQLite ``turns`` and ``model_calls`` rows
is captured on the returned ``TurnTelemetry`` so the runner doesn't have
to peek inside.
"""

from __future__ import annotations

from typing import Literal

from cot_knob.agents.base import Agent, TurnTelemetry
from cot_knob.games.base import GameState
from cot_knob.games.reversi import ReversiState
from cot_knob.llm.client import LLMClient
from cot_knob.memory.base import MemoryManager
from cot_knob.prompts.reversi import (
    SYSTEM_PROMPT,
    PromptVariant,
    render_reason_prompt,
    render_select_prompt,
)


class LLMAgent(Agent):
    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        memory: MemoryManager,
        *,
        budget: int,
        prompt_variant: PromptVariant = "free_cot",
        temperature: float = 0.0,
        side: Literal[1, -1] = 1,
    ) -> None:
        self._client = client
        self._memory = memory
        self.budget = int(budget)
        self.prompt_variant: PromptVariant = prompt_variant
        self.temperature = float(temperature)
        self.side = side

    async def choose(self, state: GameState, *, seed: int | None = None) -> TurnTelemetry:
        if not isinstance(state, ReversiState):
            raise TypeError(f"LLMAgent currently supports ReversiState only, got {type(state)}")

        legal = state.legal_moves()
        if not legal:
            return TurnTelemetry(
                chosen_move=-1,
                legal_moves=[],
                state_serialized=state.to_serializable(),
                extra={"agent": "llm", "passed": True, "budget": self.budget},
            )

        # 1. Render memory.
        mem = await self._memory.render()

        # 2. Pass 1: free reasoning at budget B.
        reason_prompt = render_reason_prompt(
            state, memory_text=mem.text, variant=self.prompt_variant
        )
        if self.budget > 0:
            comp = await self._client.generate(
                reason_prompt,
                max_tokens=self.budget,
                temperature=self.temperature,
                seed=seed,
                system=SYSTEM_PROMPT,
            )
            pass1_text = comp.text
            pass1_tokens_in = comp.n_input_tokens
            pass1_tokens_out = comp.n_output_tokens
            pass1_finish = comp.finish_reason
            pass1_latency = comp.latency_ms
        else:
            # B=0 cell: skip Pass 1 entirely.
            pass1_text = ""
            pass1_tokens_in = 0
            pass1_tokens_out = 0
            pass1_finish = "skipped"
            pass1_latency = 0.0

        # 3. Pass 2: constrained selection.
        select_prompt, choices = render_select_prompt(
            state, reason_prompt=reason_prompt, pass1_text=pass1_text
        )
        choice = await self._client.generate_choice(
            select_prompt,
            choices=choices,
            temperature=self.temperature,
            seed=seed,
            system=SYSTEM_PROMPT,
        )
        parse_failed = bool(choice.raw.get("__choice_parse_failed", False))

        chosen_str = choice.choice_text
        chosen_move = state.str_to_move(chosen_str)
        if chosen_move is None or chosen_move not in legal:
            # Defensive fallback: pick first legal. Marked as parse failure.
            chosen_move = legal[0]
            parse_failed = True

        return TurnTelemetry(
            chosen_move=chosen_move,
            legal_moves=legal,
            state_serialized=state.to_serializable(),
            llm_pass1_text=pass1_text,
            llm_pass1_tokens_in=pass1_tokens_in,
            llm_pass1_tokens_out=pass1_tokens_out,
            llm_pass1_finish=pass1_finish,
            llm_pass1_latency_ms=pass1_latency,
            llm_pass2_choice_text=chosen_str,
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
