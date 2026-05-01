"""SQLite ``Store`` for experiment results.

Single source of truth. Every write happens inside an explicit transaction
so a crashed run leaves the DB consistent.

Schema lives in [`schema.sql`](schema.sql) — see also `docs/tracking_schema.md`.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _new_id(prefix: str = "") -> str:
    suffix = uuid.uuid4().hex[:12]
    return f"{prefix}{suffix}" if prefix else suffix


class Store:
    """Thin sqlite3 wrapper. Not thread-safe across coroutines that write
    concurrently — for the smoke sweep we run sequentially. Cross-game
    batching (proposal Task 2) will need a writer queue or a coarser lock.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # --- schema management ----------------------------------------------------

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text()
        # executescript runs the whole script in its own (auto-committed)
        # transactions; do NOT wrap it in our `_tx` helper.
        cur = self._conn.cursor()
        cur.executescript(sql)
        row = cur.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] if row and row["v"] is not None else 0
        if current < SCHEMA_VERSION:
            with self._tx() as tx:
                tx.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, _now()),
                )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            yield cur
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._conn.close()

    # --- runs -----------------------------------------------------------------

    def insert_run(
        self,
        *,
        name: str,
        config_hash: str,
        config_yaml: str,
        git_sha: str | None = None,
        host: str | None = None,
        gpu: str | None = None,
        backend: str | None = None,
        model: str | None = None,
    ) -> str:
        run_id = _new_id("run_")
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO runs(run_id, name, started_at, git_sha, config_hash,
                                 config_yaml, host, gpu, backend, model, status)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (run_id, name, _now(), git_sha, config_hash, config_yaml,
                 host, gpu, backend, model),
            )
        return run_id

    def finalize_run(self, run_id: str, *, status: str = "done") -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE runs SET finished_at=?, status=? WHERE run_id=?",
                (_now(), status, run_id),
            )

    # --- trials ---------------------------------------------------------------

    def insert_trial(
        self,
        *,
        run_id: str,
        cell_index: int,
        condition: dict[str, Any],
        seed: int,
        llm_side: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        trial_id = _new_id("trial_")
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO trials(trial_id, run_id, cell_index, condition_json,
                                   seed, llm_side, started_at, extra_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (trial_id, run_id, cell_index, json.dumps(condition, sort_keys=True),
                 seed, llm_side, _now(),
                 json.dumps(extra or {}, sort_keys=True)),
            )
        return trial_id

    def finalize_trial(
        self,
        trial_id: str,
        *,
        winner: str,
        n_turns_total: int,
        n_llm_turns: int,
        n_uct_turns: int,
        final_score_llm: int,
        final_score_uct: int,
        error: str | None = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                UPDATE trials
                   SET finished_at=?, winner=?, n_turns_total=?, n_llm_turns=?,
                       n_uct_turns=?, final_score_llm=?, final_score_uct=?, error=?
                 WHERE trial_id=?
                """,
                (_now(), winner, n_turns_total, n_llm_turns, n_uct_turns,
                 final_score_llm, final_score_uct, error, trial_id),
            )

    # --- turns / model_calls / summaries -------------------------------------

    def insert_turn(
        self,
        *,
        trial_id: str,
        turn_idx: int,
        phase: str,
        player: int,
        agent_kind: str,
        board_state: dict[str, Any],
        legal_moves: list[str],
        chosen_move: str,
        chosen_move_id: int | None,
        uct_top3: list[dict[str, Any]] | None,
        move_quality: int | None,
        latency_ms_total: float,
        parse_failed: bool = False,
    ) -> str:
        turn_id = _new_id("turn_")
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO turns(turn_id, trial_id, turn_idx, phase, player, agent_kind,
                                  board_state_json, legal_moves_json, chosen_move,
                                  chosen_move_id, uct_top3_json, move_quality,
                                  latency_ms_total, parse_failed)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (turn_id, trial_id, turn_idx, phase, player, agent_kind,
                 json.dumps(board_state, sort_keys=True),
                 json.dumps(legal_moves),
                 chosen_move, chosen_move_id,
                 json.dumps(uct_top3) if uct_top3 is not None else None,
                 move_quality, float(latency_ms_total), int(parse_failed)),
            )
        return turn_id

    def insert_model_call(
        self,
        *,
        trial_id: str,
        turn_id: str | None,
        role: str,
        prompt_text: str,
        response_text: str,
        n_input_tokens: int,
        n_output_tokens: int,
        finish_reason: str | None,
        temperature: float | None,
        seed: int | None,
        latency_ms: float,
        backend: str | None,
        model: str | None,
    ) -> str:
        call_id = _new_id("call_")
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO model_calls(call_id, turn_id, trial_id, role, prompt_text,
                                        response_text, n_input_tokens, n_output_tokens,
                                        finish_reason, temperature, seed, latency_ms,
                                        backend, model)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (call_id, turn_id, trial_id, role, prompt_text, response_text,
                 int(n_input_tokens), int(n_output_tokens),
                 finish_reason, temperature, seed, float(latency_ms),
                 backend, model),
            )
        return call_id

    def insert_summary(
        self,
        *,
        trial_id: str,
        after_turn_idx: int,
        summary_text: str,
        n_tokens_est: int,
        kind: str,
    ) -> str:
        summary_id = _new_id("sum_")
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO summaries(summary_id, trial_id, after_turn_idx,
                                      summary_text, n_tokens_est, kind)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (summary_id, trial_id, after_turn_idx, summary_text,
                 int(n_tokens_est), kind),
            )
        return summary_id

    # --- read helpers ---------------------------------------------------------

    def trial_count(self, run_id: str) -> int:
        cur = self._conn.cursor()
        row = cur.execute("SELECT COUNT(*) AS c FROM trials WHERE run_id=?", (run_id,)).fetchone()
        return int(row["c"])

    def fetch_run_trials(self, run_id: str) -> list[sqlite3.Row]:
        cur = self._conn.cursor()
        return list(
            cur.execute("SELECT * FROM trials WHERE run_id=? ORDER BY cell_index", (run_id,))
        )
