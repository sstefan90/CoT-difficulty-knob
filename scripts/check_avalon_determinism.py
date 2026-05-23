"""Determinism spot-check for Avalon runs.

Runs the same single game twice with identical seed and config, then diffs
the two JSONL traces field-by-field. Any divergence at temp=0 indicates
non-determinism in the inference backend (SGLang batch-size sensitivity,
driver differences, etc.).

Usage:
    uv run python scripts/check_avalon_determinism.py configs/avalon_pilot_servant_n10.yaml

The script exits 0 if traces match and 1 if they diverge (with a diff printed).

Why this matters: temp=0 on SGLang is not guaranteed bit-stable across GPU
configurations or batch sizes. If "deterministic" runs are actually stochastic,
replications are invalid and CI intervals are wrong. Run this once per new
hardware setup or SGLang version upgrade.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml

from cot_knob.experiments.avalon_sweep import AvalonSweepConfig
from cot_knob.games.avalon_runner import make_env_from_seed, run_avalon_game
from cot_knob.llm.factory import build_client
from cot_knob.tracking.jsonl import JSONLWriter
from cot_knob.tracking.store import Store


async def _run_one(cfg: AvalonSweepConfig, seed: int, tmp_dir: Path, tag: str) -> list[dict]:
    db_path = tmp_dir / f"det_{tag}.db"
    run_name = f"det_{tag}"

    store = Store(db_path)
    budget = cfg.budgets[0]
    run_id = store.insert_run(
        name=run_name,
        config_hash="det",
        config_yaml=yaml.dump(cfg.model_dump()),
    )
    jsonl = JSONLWriter(tmp_dir / "runs", run_id, run_name=run_name)
    trial_id = store.insert_trial(
        run_id=run_id, cell_index=0,
        condition={"budget": budget, "seed": seed, "role": cfg.llm_role},
        seed=seed, llm_side=0,
    )

    env = make_env_from_seed(seed, cfg.llm_role)
    client = build_client(cfg.model.model_dump())

    await run_avalon_game(
        env=env,
        llm_player_idx=0,
        llm_role=cfg.llm_role,
        llm_client=client,
        budget=budget,
        prompt_variant=cfg.prompt_variant,
        store=store,
        jsonl=jsonl,
        run_id=run_id,
        trial_id=trial_id,
        condition={"budget": budget, "seed": seed},
        seed=seed,
        temperature=cfg.model.temperature,
        backend=cfg.model.backend,
        model=cfg.model.name,
    )

    store.close()
    jsonl.close()

    # Per-trial file lives at runs/{run_name}__{short_id}/{trial_id}.jsonl
    short_id = run_id.replace("run_", "")[:8]
    jl_path = tmp_dir / "runs" / f"{run_name}__{short_id}" / f"{trial_id}.jsonl"
    events = []
    with open(jl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# Fields that are expected to differ across runs and don't indicate non-determinism.
_NOISE_KEYS = frozenset({
    "ts",
    "trial_id",
    "llm_pass1_latency_ms",
    "llm_pass2_latency_ms",
    "llm_disc_latency_ms",
})


def _diff_traces(a: list[dict], b: list[dict]) -> list[str]:
    diffs: list[str] = []
    if len(a) != len(b):
        diffs.append(f"Event count differs: run_a={len(a)}, run_b={len(b)}")
        n = min(len(a), len(b))
    else:
        n = len(a)

    for i, (ea, eb) in enumerate(zip(a[:n], b[:n])):
        all_keys = set(ea) | set(eb)
        diff_keys = [k for k in all_keys if ea.get(k) != eb.get(k) and k not in _NOISE_KEYS]
        if diff_keys:
            diffs.append(
                f"Event {i} ({ea.get('event', '?')}) differs on: {diff_keys}\n"
                f"  run_a: { {k: ea.get(k) for k in diff_keys} }\n"
                f"  run_b: { {k: eb.get(k) for k in diff_keys} }"
            )
    return diffs


async def main() -> int:
    parser = argparse.ArgumentParser(description="Avalon determinism check")
    parser.add_argument("config", help="Path to avalon sweep YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for both runs")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = AvalonSweepConfig.model_validate(raw)

    print(f"Running game twice with seed={args.seed}, budget={cfg.budgets[0]}, role={cfg.llm_role} ...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        trace_a = await _run_one(cfg, args.seed, tmp_dir, "a")
        trace_b = await _run_one(cfg, args.seed, tmp_dir, "b")

    diffs = _diff_traces(trace_a, trace_b)

    if not diffs:
        print("✓  Traces are identical — backend is deterministic for this seed.")
        return 0
    else:
        print(f"✗  {len(diffs)} difference(s) found — backend is NOT deterministic!\n")
        for d in diffs:
            print(d)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
