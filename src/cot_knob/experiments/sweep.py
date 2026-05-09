"""Run a budget sweep defined by a SweepConfig.

For the smoke run: 4 budgets × N=3 seeds = 12 games against UCT-2000.
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
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cot_knob.agents.llm_agent import LLMAgent
from cot_knob.agents.nim_optimal import NimOptimalAgent
from cot_knob.agents.random_agent import RandomAgent
from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.experiments.config import SweepConfig
from cot_knob.experiments.runner import play_match
from cot_knob.llm.factory import build_client
from cot_knob.memory.full_history import FullHistoryMemory
from cot_knob.memory.last_move import LastMoveMemory
from cot_knob.memory.summary import StructuredSummaryMemory
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store

console = Console()


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _config_hash(yaml_text: str) -> str:
    return hashlib.sha256(yaml_text.encode()).hexdigest()[:16]


async def run_sweep(
    cfg: SweepConfig,
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
    console.rule(f"[bold cyan]Sweep '{cfg.run_name}' (run_id={run_id})")
    console.print(f"  backend={cfg.model.backend}  model={cfg.model.name}")
    oracle_label = (
        "NimOptimal" if cfg.game == "nim"
        else f"UCT-{cfg.oracle_iterations}"
    )
    opp_label = (
        f"UCT-{cfg.opponent.iterations}" if cfg.opponent.kind == "uct"
        else cfg.opponent.kind
    )
    console.print(
        f"  game={cfg.game}  budgets={cfg.budgets}  N/cell={cfg.n_per_cell}  "
        f"opponent={opp_label}  oracle={oracle_label}"
    )

    seeds = cfg.resolved_seeds()

    # Build (cell_index, budget, seed, llm_side) tuples.
    # When llm_plays="both", alternate sides so the first half of seeds play as
    # player 1 and the second half as player 2 (evenly split).
    cells: list[tuple[int, int, int, int]] = []
    cell_index = 0
    if cfg.llm_plays == "both":
        sides_cycle = [1, -1]
        for B in cfg.budgets:
            for i, seed in enumerate(seeds):
                side = sides_cycle[i % 2]
                cells.append((cell_index, B, seed, side))
                cell_index += 1
    else:
        fixed_side = 1 if cfg.llm_plays == "black" else -1
        for B in cfg.budgets:
            for seed in seeds:
                cells.append((cell_index, B, seed, fixed_side))
                cell_index += 1

    # ── Game-specific setup ───────────────────────────────────────────────────
    if cfg.game == "nim":
        from cot_knob.games.nim import initial_state as nim_initial_state
        nim_piles = tuple(cfg.nim_piles)

        def make_initial_state():
            return nim_initial_state(nim_piles)

        def make_oracle():
            return NimOptimalAgent(top_k=None)

        def make_opponent(seed_val: int, llm_agent_ref: LLMAgent | None = None):
            if cfg.opponent.kind == "nim_optimal":
                return NimOptimalAgent(seed=seed_val)
            elif cfg.opponent.kind == "random":
                return RandomAgent(seed=seed_val)
            elif cfg.opponent.kind == "llm":
                # Self-play: build a second LLMAgent with opp_budget.
                # Memory for the opponent is always last_move (cheap).
                opp_mem = LastMoveMemory()
                return LLMAgent(
                    client, opp_mem,
                    budget=cfg.opponent.opp_budget,
                    game=cfg.game,
                    prompt_variant=cfg.prompt_variant,
                    temperature=cfg.model.temperature,
                    side=-1,  # opponent always plays as player 2
                )
            return NimOptimalAgent(seed=seed_val)  # default for Nim

        opponent_label = (
            f"llm-B{cfg.opponent.opp_budget}"
            if cfg.opponent.kind == "llm"
            else cfg.opponent.kind
        )
    else:
        from cot_knob.games.reversi import initial_state as rev_initial_state

        def make_initial_state():
            return rev_initial_state()

        def make_oracle():
            return UCTAgent(iterations=cfg.oracle_iterations, top_k=None)

        def make_opponent(seed_val: int, llm_agent_ref: LLMAgent | None = None):
            if cfg.opponent.kind == "random":
                return RandomAgent(seed=seed_val)
            return UCTAgent(iterations=cfg.opponent.iterations, seed=seed_val)

        opponent_label = (
            f"uct-{cfg.opponent.iterations}"
            if cfg.opponent.kind == "uct"
            else cfg.opponent.kind
        )

    client = build_client(cfg.model.model_dump())
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("trials", total=len(cells))
            for idx, B, seed, llm_side in cells:
                if cfg.memory.kind == "structured_summary":
                    memory: Any = StructuredSummaryMemory(
                        client,
                        max_summary_tokens=cfg.memory.max_summary_tokens,
                        summarize_every=cfg.memory.summarize_every,
                        seed=seed,
                    )
                elif cfg.memory.kind == "last_move":
                    memory = LastMoveMemory()
                else:
                    memory = FullHistoryMemory()

                llm_agent = LLMAgent(
                    client, memory,
                    budget=B,
                    game=cfg.game,
                    prompt_variant=cfg.prompt_variant,
                    temperature=cfg.model.temperature,
                    side=llm_side,
                )
                opp_agent = make_opponent(seed, llm_agent)
                oracle = make_oracle()
                condition = {
                    "budget": B,
                    "model": cfg.model.name,
                    "backend": cfg.model.backend,
                    "opponent": opponent_label,
                    "memory": cfg.memory.kind,
                    "prompt_variant": cfg.prompt_variant,
                    "temperature": cfg.model.temperature,
                    "llm_side": "black" if llm_side == 1 else "white",
                    "seed": seed,
                    "game": cfg.game,
                }
                bar.update(task, description=f"B={B:>4} seed={seed} side={'p1' if llm_side==1 else 'p2'}")
                result = await play_match(
                    store=store, jsonl=jsonl, run_id=run_id,
                    cell_index=idx, condition=condition,
                    seed=seed, llm_side=llm_side,
                    llm_agent=llm_agent, uct_agent=opp_agent,
                    opp_kind=opponent_label,
                    oracle_iters=cfg.oracle_iterations,
                    oracle_agent=oracle,
                    initial=make_initial_state(),
                    max_turns_safety=cfg.max_turns_safety,
                    early_cutoff=cfg.early_cutoff,
                    late_after=cfg.late_after,
                    backend=cfg.model.backend, model=cfg.model.name,
                )
                bar.update(task, advance=1)
                console.log(
                    f"  cell={idx:>2} B={B:>4} seed={seed} side={'p1' if llm_side==1 else 'p2'} -> "
                    f"winner={result.winner} ({result.final_score_llm}-{result.final_score_uct}) "
                    f"in {result.n_turns} turns"
                )
        store.finalize_run(run_id)
    except Exception:
        store.finalize_run(run_id, status="error")
        raise
    finally:
        await client.aclose()
        jsonl.close()
        store.close()

    console.rule("[bold green]Sweep complete")
    console.print(f"run_id={run_id}")
    return run_id


def main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run a CoT-budget sweep.")
    p.add_argument("config", help="Path to a YAML SweepConfig.")
    p.add_argument(
        "--db",
        default=os.environ.get("RESULTS_DB", "data/results.db"),
        help="SQLite DB path (default: data/results.db).",
    )
    p.add_argument(
        "--runs",
        default=os.environ.get("RUNS_DIR", "data/runs"),
        help="Runs directory for JSONL backups (default: data/runs).",
    )
    args = p.parse_args()

    from cot_knob.experiments.config import load_sweep_config

    cfg = load_sweep_config(args.config)
    asyncio.run(run_sweep(cfg, db_path=args.db, runs_dir=args.runs))


if __name__ == "__main__":  # pragma: no cover
    main_cli()
