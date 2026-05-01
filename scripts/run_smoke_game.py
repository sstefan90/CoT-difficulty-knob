"""Run a single Reversi self-play game (LLM vs UCT) for sanity-checking.

Useful for verifying the harness without committing to a full sweep::

    uv run python scripts/run_smoke_game.py --backend mock
    uv run python scripts/run_smoke_game.py --backend ollama --budget 256
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console

from cot_knob.agents.llm_agent import LLMAgent
from cot_knob.agents.uct_agent import UCTAgent
from cot_knob.experiments.runner import play_match
from cot_knob.llm.factory import build_client
from cot_knob.memory.full_history import FullHistoryMemory
from cot_knob.memory.summary import StructuredSummaryMemory
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store

console = Console()


async def main_async(args: argparse.Namespace) -> None:
    client = build_client({
        "backend": args.backend,
        "name": args.model,
        "base_url": args.base_url,
        "temperature": 0.0,
    })
    if args.memory == "structured_summary":
        memory = StructuredSummaryMemory(
            client, max_summary_tokens=args.summary_tokens, summarize_every=4, seed=args.seed
        )
    else:
        memory = FullHistoryMemory()

    llm_agent = LLMAgent(client, memory, budget=args.budget, side=1)
    uct = UCTAgent(iterations=args.uct_iterations, seed=args.seed)

    store = Store(args.db)
    jsonl = JSONLWriter(args.runs, "smoke_one")
    run_id = store.insert_run(
        name="smoke_one_game", config_hash="adhoc", config_yaml="(smoke one-game)",
        backend=args.backend, model=args.model,
    )
    try:
        result = await play_match(
            store=store, jsonl=jsonl, run_id=run_id, cell_index=0,
            condition={
                "budget": args.budget, "model": args.model, "backend": args.backend,
                "opponent": f"uct-{args.uct_iterations}", "memory": args.memory,
                "seed": args.seed, "llm_side": "black",
            },
            seed=args.seed, llm_side=1,
            llm_agent=llm_agent, uct_agent=uct,
            backend=args.backend, model=args.model,
        )
        store.finalize_run(run_id)
        console.rule("[bold green]Done")
        console.print(result)
    finally:
        await client.aclose()
        jsonl.close()
        store.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock", choices=["mock", "ollama", "sglang"])
    p.add_argument("--model", default="mock-r1")
    p.add_argument("--base-url", default=None)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--memory", default="structured_summary",
                   choices=["structured_summary", "full_history"])
    p.add_argument("--summary-tokens", type=int, default=128)
    p.add_argument("--uct-iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--db", default="data/results.db")
    p.add_argument("--runs", default="data/runs")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
