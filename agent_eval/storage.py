"""SQLite persistence for runs, attempts, feedback and results.

Measurement and regression detection are impossible without history. SQLite
is zero-setup and file-based -- the right weight for this project. All
queries are parameterised; per-test results are stored as JSON text.

The DB path is read from ``config.DB_PATH`` at call time so tests can point
it at a temp file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_eval import config
from agent_eval.models import Attempt, RunInfo, Task, TaskResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    feedback_level  TEXT NOT NULL DEFAULT 'full',
    max_tries       INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    suite_hash      TEXT NOT NULL DEFAULT '',
    seed            INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'running',
    finished_at     TEXT,
    suite           TEXT NOT NULL DEFAULT 'easy'
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    prompt      TEXT NOT NULL,
    entry_point TEXT NOT NULL,
    test_code   TEXT NOT NULL,
    difficulty  TEXT NOT NULL DEFAULT 'medium',
    tags        TEXT NOT NULL DEFAULT '[]',
    category    TEXT NOT NULL DEFAULT 'function'
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task_id           TEXT NOT NULL REFERENCES tasks(task_id),
    attempt_number    INTEGER NOT NULL,
    code              TEXT NOT NULL,
    test_results      TEXT NOT NULL,
    passed            INTEGER NOT NULL,
    error_type        TEXT,
    num_passed        INTEGER NOT NULL,
    num_total         INTEGER NOT NULL,
    feedback_given    TEXT,
    feedback_produced TEXT,
    raw_response      TEXT NOT NULL DEFAULT '',
    tokens_in         INTEGER NOT NULL DEFAULT 0,
    tokens_out        INTEGER NOT NULL DEFAULT 0,
    latency_s         REAL NOT NULL DEFAULT 0,
    sandbox_s         REAL NOT NULL DEFAULT 0,
    UNIQUE (run_id, task_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS results (
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task_id       TEXT NOT NULL REFERENCES tasks(task_id),
    passed        INTEGER NOT NULL,
    tries_taken   INTEGER NOT NULL,
    final_code    TEXT NOT NULL,
    quality_score TEXT,
    code_metrics  TEXT,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    latency_s     REAL NOT NULL DEFAULT 0,
    error         TEXT,
    PRIMARY KEY (run_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_attempts_run_task ON attempts(run_id, task_id);
"""


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the first schema version: (table, column, DDL). init_db adds
# any that an older database file is missing, so existing run history keeps working.
MIGRATIONS = (
    ("runs", "suite", "TEXT NOT NULL DEFAULT 'easy'"),
    ("tasks", "category", "TEXT NOT NULL DEFAULT 'function'"),
)


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        for table, column, ddl in MIGRATIONS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    conn.close()


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value)


def _loads(value):
    return None if value is None else json.loads(value)


def create_run(run: RunInfo) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, model, prompt_version, feedback_level, max_tries, timestamp,"
            " suite_hash, seed, notes, suite) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.run_id, run.model, run.prompt_version, run.feedback_level, run.max_tries,
             run.timestamp, run.suite_hash, run.seed, run.notes, run.suite),
        )
    conn.close()


def finish_run(run_id: str, finished_at: str, status: str = "completed") -> None:
    with get_connection() as conn:
        conn.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                     (status, finished_at, run_id))
    conn.close()


def _save_task(conn: sqlite3.Connection, task: Task) -> None:
    conn.execute(
        "INSERT INTO tasks (task_id, prompt, entry_point, test_code, difficulty, tags, category)"
        " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET prompt = excluded.prompt,"
        " entry_point = excluded.entry_point, test_code = excluded.test_code,"
        " difficulty = excluded.difficulty, tags = excluded.tags, category = excluded.category",
        (task.task_id, task.prompt, task.entry_point, task.test_code, task.difficulty,
         json.dumps(task.tags), task.category),
    )


def _save_attempt(conn: sqlite3.Connection, run_id: str, task_id: str, attempt: Attempt) -> None:
    tr = attempt.test_result
    conn.execute(
        "INSERT INTO attempts (run_id, task_id, attempt_number, code, test_results, passed,"
        " error_type, num_passed, num_total, feedback_given, feedback_produced, raw_response,"
        " tokens_in, tokens_out, latency_s, sandbox_s)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, task_id, attempt.attempt_number, attempt.code, json.dumps(tr.to_dict()),
         int(tr.passed), tr.primary_error_type, tr.num_passed, tr.num_total,
         attempt.feedback_given, attempt.feedback_produced, attempt.raw_response,
         attempt.tokens_in, attempt.tokens_out, attempt.latency_s, tr.duration_s),
    )


def _save_result(conn: sqlite3.Connection, run_id: str, result: TaskResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (run_id, task_id, passed, tries_taken, final_code,"
        " quality_score, code_metrics, tokens_in, tokens_out, latency_s, error)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, result.task_id, int(result.passed), result.tries_taken, result.final_code,
         _dumps(result.quality_score), _dumps(result.code_metrics), result.total_tokens_in,
         result.total_tokens_out, result.total_latency_s, result.error),
    )


def save_task(task: Task) -> None:
    with get_connection() as conn:
        _save_task(conn, task)
    conn.close()


def save_attempt(run_id: str, task_id: str, attempt: Attempt) -> None:
    with get_connection() as conn:
        _save_attempt(conn, run_id, task_id, attempt)
    conn.close()


def save_result(run_id: str, result: TaskResult) -> None:
    with get_connection() as conn:
        _save_result(conn, run_id, result)
    conn.close()


def save_task_run(run_id: str, task: Task, result: TaskResult, attempts: list[Attempt]) -> None:
    """Persist one task's full outcome atomically."""
    with get_connection() as conn:
        _save_task(conn, task)
        for attempt in attempts:
            _save_attempt(conn, run_id, task.task_id, attempt)
        _save_result(conn, run_id, result)
    conn.close()


# --- reads -----------------------------------------------------------------

def list_runs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT r.*, COUNT(res.task_id) AS num_tasks, COALESCE(SUM(res.passed), 0) AS num_passed"
            " FROM runs r LEFT JOIN results res ON res.run_id = r.run_id"
            " GROUP BY r.run_id ORDER BY r.timestamp DESC"
        ).fetchall()
    conn.close()
    out = []
    for row in rows:
        d = dict(row)
        d["pass_rate"] = d["num_passed"] / d["num_tasks"] if d["num_tasks"] else None
        out.append(d)
    return out


def get_run(run_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def resolve_run_id(ref: str) -> str:
    """Accept a full run id, a unique prefix, 'latest', or 'latest~N'."""
    runs = list_runs()
    if ref.startswith("latest"):
        offset = int(ref.split("~", 1)[1]) if "~" in ref else 0
        if offset >= len(runs):
            raise KeyError(f"only {len(runs)} runs stored; cannot resolve {ref!r}")
        return runs[offset]["run_id"]
    matches = [r["run_id"] for r in runs if r["run_id"].startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no run matches {ref!r}")
    raise KeyError(f"{ref!r} is ambiguous ({len(matches)} runs match)")


def resolve_run_refs(ref: str) -> list[str]:
    """Resolve a run reference that may name several runs:
    comma-separated ids/prefixes, or a config selector ``@model[/feedback[/prompt[/suite]]]``
    (all stored runs of that configuration, e.g. every seed)."""
    if ref.startswith("@"):
        parts = ref[1:].split("/")
        keys = ("model", "feedback_level", "prompt_version", "suite")
        wanted = dict(zip(keys, parts, strict=False))
        matches = [r["run_id"] for r in list_runs()
                   if all(r[k] == v for k, v in wanted.items() if v)]
        if not matches:
            raise KeyError(f"no runs match selector {ref!r}")
        return matches
    return [resolve_run_id(part.strip()) for part in ref.split(",") if part.strip()]


def get_results(run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT res.*, t.difficulty, t.tags, t.category FROM results res"
            " LEFT JOIN tasks t ON t.task_id = res.task_id"
            " WHERE res.run_id = ? ORDER BY res.task_id",
            (run_id,),
        ).fetchall()
    conn.close()
    out = []
    for row in rows:
        d = dict(row)
        d["passed"] = bool(d["passed"])
        d["quality_score"] = _loads(d["quality_score"])
        d["code_metrics"] = _loads(d["code_metrics"])
        d["tags"] = _loads(d["tags"]) or []
        out.append(d)
    return out


def get_attempts(run_id: str, task_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM attempts WHERE run_id = ?"
    params: tuple = (run_id,)
    if task_id is not None:
        query += " AND task_id = ?"
        params += (task_id,)
    query += " ORDER BY task_id, attempt_number"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    conn.close()
    out = []
    for row in rows:
        d = dict(row)
        d["passed"] = bool(d["passed"])
        d["test_results"] = json.loads(d["test_results"])
        out.append(d)
    return out


def get_task(task_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["tags"] = json.loads(d["tags"])
    return d


def delete_run(run_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM attempts WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.close()
