"""Avalon experiment sweep runner.

``run_avalon_sweep()`` executes a cross-product of (budget, seed) cells
defined in an ``AvalonSweepConfig`` YAML. Each cell runs one game with the
LLM as a fixed role against four naive bots.

Usage:
    python scripts/run_avalon_sweep.py configs/avalon_pilot_servant_n10.yaml
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cot_knob.games.avalon_runner import AvalonGameResult, make_env_from_seed, run_avalon_game
from cot_knob.llm.factory import build_client
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store

console = Console()


# ── Config model ──────────────────────────────────────────────────────────

class AvalonLLMConfig(BaseModel):
    backend: str = "sglang"
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    base_url: str | None = None
    temperature: float = 0.0
    think: bool = False
    extra: dict = Field(default_factory=dict)


class AvalonSweepConfig(BaseModel):
    run_name: str
    model: AvalonLLMConfig
    llm_role: str = "Servant"            # "Servant" | "Merlin"
    llm_player_idx: int = 0              # always 0 in standard setup
    prompt_variant: str = "minimal"      # "minimal" | "procedural"
    budgets: list[int] = Field(default_factory=lambda: [64, 1024])
    n_per_cell: int = 10
    seeds: list[int] | None = None
    with_discussion: bool = False        # enable discussion phase before team votes
    with_summarizer: bool = False        # call Sonnet after each quest to summarize
    summarizer_model: str = "claude-sonnet-4-6"  # model for post-quest summaries
    summarizer_backend: str = "anthropic"          # "anthropic" | "sglang"
    summarizer_base_url: str | None = None         # SGLang base URL (sglang backend only)
    shuffle_all_roles: bool = False      # shuffle ALL 4 non-LLM roles together (evil not pinned to P3/P4)

    @field_validator("llm_role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        if v not in ("Servant", "Merlin"):
            raise ValueError(f"llm_role must be 'Servant' or 'Merlin', got {v!r}")
        return v

    @field_validator("budgets")
    @classmethod
    def _valid_budgets(cls, v: list[int]) -> list[int]:
        if not v or any(b < 0 for b in v):
            raise ValueError("budgets must be non-empty and non-negative")
        return v

    def resolved_seeds(self) -> list[int]:
        if self.seeds is not None and len(self.seeds) >= self.n_per_cell:
            return list(self.seeds[: self.n_per_cell])
        base = self.seeds or list(range(self.n_per_cell))
        out = list(base)
        i = 0
        while len(out) < self.n_per_cell:
            out.append(base[i % len(base)] + 1000 * (len(out) // max(1, len(base))))
            i += 1
        return out[:self.n_per_cell]


def load_avalon_sweep_config(path: str) -> AvalonSweepConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AvalonSweepConfig.model_validate(data)


# ── Sweep runner ───────────────────────────────────────────────────────────

def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _config_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def run_avalon_sweep(
    cfg: AvalonSweepConfig,
    *,
    db_path: str | Path,
    runs_dir: str | Path,
) -> str:
    yaml_text = yaml.safe_dump(cfg.model_dump(), sort_keys=True)
    store = Store(db_path)

    run_id = store.insert_run(
        name=cfg.run_name,
        config_hash=_config_hash(yaml_text),
        config_yaml=yaml_text,
        git_sha=_git_sha(),
        host=socket.gethostname(),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES") or platform.machine(),
        backend=cfg.model.backend,
        model=cfg.model.name,
    )
    jsonl = JSONLWriter(runs_dir, run_id, run_name=cfg.run_name)

    console.rule(f"[bold cyan]Avalon sweep '{cfg.run_name}' (run_id={run_id})")
    console.print(
        f"  role={cfg.llm_role}  prompt={cfg.prompt_variant}  "
        f"budgets={cfg.budgets}  N/cell={cfg.n_per_cell}  "
        f"discussion={cfg.with_discussion}  summarizer={cfg.with_summarizer}  "
        f"backend={cfg.model.backend}  model={cfg.model.name}"
    )

    seeds = cfg.resolved_seeds()
    cells: list[tuple[int, int, int]] = []
    cell_idx = 0
    for B in cfg.budgets:
        for s in seeds:
            cells.append((cell_idx, B, s))
            cell_idx += 1

    client_cfg = cfg.model.model_dump()
    client = build_client(client_cfg)

    summarizer = None
    if cfg.with_summarizer:
        from cot_knob.games.avalon_summarizer import AvalonSummarizer, AvalonSummarizerSGLang
        if cfg.summarizer_backend == "sglang":
            summarizer = AvalonSummarizerSGLang(
                model=cfg.summarizer_model,
                base_url=cfg.summarizer_base_url or "http://localhost:30000",
            )
        else:
            summarizer = AvalonSummarizer(model=cfg.summarizer_model)

    wins = 0
    total = 0
    results: list[AvalonGameResult] = []

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("games", total=len(cells))

            for idx, B, seed in cells:
                condition: dict[str, Any] = {
                    "game": "avalon",
                    "llm_role": cfg.llm_role,
                    "budget": B,
                    "prompt_variant": cfg.prompt_variant,
                    "with_discussion": cfg.with_discussion,
                    "model": cfg.model.name,
                    "backend": cfg.model.backend,
                    "temperature": cfg.model.temperature,
                    "seed": seed,
                }
                trial_id = store.insert_trial(
                    run_id=run_id,
                    cell_index=idx,
                    condition=condition,
                    seed=seed,
                    llm_side=cfg.llm_player_idx,
                )
                jsonl.write(trial_id, "trial_start", {"trial_id": trial_id, "condition": condition})

                env = make_env_from_seed(seed, cfg.llm_role, cfg.llm_player_idx,
                                         shuffle_all_roles=cfg.shuffle_all_roles)

                bar.update(task, description=f"B={B:>4} seed={seed:>4} role={cfg.llm_role}")

                result = await run_avalon_game(
                    env=env,
                    llm_player_idx=cfg.llm_player_idx,
                    llm_role=cfg.llm_role,
                    llm_client=client,
                    budget=B,
                    prompt_variant=cfg.prompt_variant,
                    store=store,
                    jsonl=jsonl,
                    run_id=run_id,
                    trial_id=trial_id,
                    condition=condition,
                    seed=seed,
                    temperature=cfg.model.temperature,
                    backend=cfg.model.backend,
                    model=cfg.model.name,
                    with_discussion=cfg.with_discussion,
                    summarizer=summarizer,
                )

                results.append(result)
                bar.update(task, advance=1)

                if not result.error:
                    total += 1
                    if result.llm_wins:
                        wins += 1
                win_rate = wins / total if total else float("nan")

                console.log(
                    f"  cell={idx:>2} B={B:>4} seed={seed:>4} "
                    f"-> {'LLM wins' if result.llm_wins else 'bots win'} "
                    f"(quests: good={result.n_good_quests}/{result.n_quests_played}, "
                    f"parse_fail={result.n_parse_failed}/{result.n_llm_decisions}) "
                    f"running_wr={win_rate:.2%}"
                )

        store.finalize_run(run_id)
    except Exception:
        store.finalize_run(run_id, status="error")
        raise
    finally:
        await client.aclose()
        if summarizer is not None:
            await summarizer.aclose()
        jsonl.close()
        store.close()

    console.rule("[bold green]Sweep complete")
    console.print(f"run_id={run_id}  final win rate={wins}/{total}={wins/total:.2%}" if total else f"run_id={run_id}")
    return run_id


def main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run an Avalon budget sweep.")
    p.add_argument("config", help="Path to YAML AvalonSweepConfig.")
    p.add_argument(
        "--db",
        default=os.environ.get("RESULTS_DB", "data/results.db"),
        help="SQLite DB path (default: data/results.db).",
    )
    p.add_argument(
        "--runs",
        default=os.environ.get("RUNS_DIR", "data/runs"),
        help="Runs directory for JSONL (default: data/runs).",
    )
    args = p.parse_args()

    cfg = load_avalon_sweep_config(args.config)
    asyncio.run(run_avalon_sweep(cfg, db_path=args.db, runs_dir=args.runs))


if __name__ == "__main__":  # pragma: no cover
    main_cli()
