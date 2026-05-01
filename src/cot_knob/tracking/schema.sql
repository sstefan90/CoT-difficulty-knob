-- CoT-knob results database. SQLite, normalized, append-only.
-- See docs/tracking_schema.md for the conceptual reference.
--
-- Schema versioning: bump SCHEMA_VERSION in store.py and add migration steps
-- there if you change anything below.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ----------------------------------------------------------------------------
-- runs: one row per run_budget_sweep.py invocation.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    started_at    TEXT NOT NULL,         -- ISO-8601 UTC
    finished_at   TEXT,                  -- ISO-8601 UTC; NULL while running
    git_sha       TEXT,
    config_hash   TEXT NOT NULL,
    config_yaml   TEXT NOT NULL,
    host          TEXT,
    gpu           TEXT,
    backend       TEXT,                  -- "ollama" | "sglang" | "mock"
    model         TEXT,
    status        TEXT NOT NULL DEFAULT 'running'  -- "running" | "done" | "error"
);

-- ----------------------------------------------------------------------------
-- trials: one row per game.
-- condition_json carries the full experimental cell so we can pivot on it.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trials (
    trial_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    cell_index      INTEGER NOT NULL,    -- ordinal within the run
    condition_json  TEXT NOT NULL,       -- {model, B, opponent, memory, prompt_variant, seed, side}
    seed            INTEGER NOT NULL,
    llm_side        INTEGER NOT NULL,    -- +1 or -1
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    n_turns_total   INTEGER,
    n_llm_turns     INTEGER,
    n_uct_turns     INTEGER,
    winner          TEXT,                -- "llm" | "uct" | "draw" | "error"
    final_score_llm INTEGER,
    final_score_uct INTEGER,
    error           TEXT,
    extra_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id);
CREATE INDEX IF NOT EXISTS idx_trials_winner ON trials(run_id, winner);

-- ----------------------------------------------------------------------------
-- turns: one row per LLM decision (NOT per UCT decision; UCT-only turns
-- are captured implicitly via the move log).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    turn_id              TEXT PRIMARY KEY,
    trial_id             TEXT NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
    turn_idx             INTEGER NOT NULL,
    phase                TEXT,                -- "early" | "mid" | "late"
    player               INTEGER NOT NULL,    -- whose turn it was
    agent_kind           TEXT NOT NULL,       -- "llm" | "uct" | "filler"
    board_state_json     TEXT NOT NULL,
    legal_moves_json     TEXT NOT NULL,
    chosen_move          TEXT NOT NULL,       -- e.g. "c4" or "pass"
    chosen_move_id       INTEGER,
    uct_top3_json        TEXT,                -- [{move, visits, win_rate}, ...]
    move_quality         INTEGER,             -- 1 if chosen ∈ uct_top3, else 0; NULL if no UCT ref
    latency_ms_total     REAL NOT NULL,
    parse_failed         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turns_trial ON turns(trial_id);
CREATE INDEX IF NOT EXISTS idx_turns_phase ON turns(trial_id, phase);

-- ----------------------------------------------------------------------------
-- model_calls: one row per HTTP request to the LLM backend.
-- role='reason' rows are the corpus the Task 9 labeler will read.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_calls (
    call_id          TEXT PRIMARY KEY,
    turn_id          TEXT REFERENCES turns(turn_id) ON DELETE CASCADE,
    trial_id         TEXT NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
    role             TEXT NOT NULL,        -- "reason" | "select" | "summarize" | "filler"
    prompt_text      TEXT NOT NULL,
    response_text    TEXT NOT NULL,
    n_input_tokens   INTEGER NOT NULL DEFAULT 0,
    n_output_tokens  INTEGER NOT NULL DEFAULT 0,
    finish_reason    TEXT,
    temperature      REAL,
    seed             INTEGER,
    latency_ms       REAL NOT NULL,
    backend          TEXT,
    model            TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_role ON model_calls(role);
CREATE INDEX IF NOT EXISTS idx_calls_trial_role ON model_calls(trial_id, role);
CREATE INDEX IF NOT EXISTS idx_calls_turn ON model_calls(turn_id);

-- ----------------------------------------------------------------------------
-- summaries: one row per memory state, written after the summarizer runs.
-- Used for the Task 5 summary-stability sub-analysis.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS summaries (
    summary_id      TEXT PRIMARY KEY,
    trial_id        TEXT NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
    after_turn_idx  INTEGER NOT NULL,
    summary_text    TEXT NOT NULL,
    n_tokens_est    INTEGER NOT NULL,
    kind            TEXT NOT NULL          -- "summary" | "full_history"
);
CREATE INDEX IF NOT EXISTS idx_summaries_trial ON summaries(trial_id);

-- ----------------------------------------------------------------------------
-- controller_decisions: empty until proposal Task 10. Reserved here so
-- adding the controller doesn't require a schema migration later.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS controller_decisions (
    decision_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id        TEXT REFERENCES trials(trial_id) ON DELETE SET NULL,
    game_index      INTEGER NOT NULL,
    window_win_rate REAL,
    chosen_b_next   INTEGER,
    notes_json      TEXT
);

-- Light schema-version table so future migrations are explicit.
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
