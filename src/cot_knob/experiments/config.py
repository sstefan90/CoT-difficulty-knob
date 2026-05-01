"""Pydantic config models for the experiment harness."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    backend: Literal["mock", "ollama", "sglang"] = "ollama"
    name: str = "deepseek-r1:7b"
    base_url: str | None = None
    temperature: float = 0.0
    extra: dict = Field(default_factory=dict)


class OpponentConfig(BaseModel):
    kind: Literal["uct"] = "uct"
    iterations: int = 2000


class MemoryConfig(BaseModel):
    kind: Literal["structured_summary", "full_history"] = "structured_summary"
    max_summary_tokens: int = 256
    summarize_every: int = 4


class SweepConfig(BaseModel):
    """Top-level config for a single sweep invocation.

    A sweep runs the cross-product of ``budgets`` × ``seeds`` (both
    repeated ``n_per_cell`` times if seeds are short).  ``llm_plays``
    fixes which side the LLM plays for the smoke run; for full sweeps
    we will alternate.
    """

    run_name: str
    model: LLMConfig
    game: Literal["reversi"] = "reversi"
    opponent: OpponentConfig
    memory: MemoryConfig
    prompt_variant: Literal["free_cot", "structured_cot"] = "free_cot"
    budgets: list[int] = Field(default_factory=lambda: [0, 64, 256, 1024])
    n_per_cell: int = 3
    seeds: list[int] | None = None
    llm_plays: Literal["black", "white"] = "black"
    max_turns_safety: int = 200

    @field_validator("budgets")
    @classmethod
    def _budgets_nonneg(cls, v: list[int]) -> list[int]:
        if not v or any(b < 0 for b in v):
            raise ValueError("budgets must be a non-empty list of non-negative ints")
        return v

    @field_validator("n_per_cell")
    @classmethod
    def _n_pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_per_cell must be >= 1")
        return v

    def resolved_seeds(self) -> list[int]:
        if self.seeds is None:
            return list(range(self.n_per_cell))
        if len(self.seeds) >= self.n_per_cell:
            return list(self.seeds[: self.n_per_cell])
        # Tile / extend as needed.
        out = list(self.seeds)
        i = 0
        while len(out) < self.n_per_cell:
            out.append(self.seeds[i % len(self.seeds)] + 1000 * (len(out) // max(1, len(self.seeds))))
            i += 1
        return out


def load_sweep_config(path: str) -> SweepConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SweepConfig.model_validate(data)
