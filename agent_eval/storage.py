# WHY THIS FILE EXISTS:
#   Build step 4: persist every run, attempt, feedback message, and result.
#   Engineering pillar #5 (measurement & regression detection) is impossible
#   without history. SQLite is zero-setup and file-based -- the right weight
#   for this project.
#
# WHAT IT NEEDS:
#   - get_connection() -> sqlite3.Connection (create data/ dir if missing).
#   - init_db(): CREATE TABLE IF NOT EXISTS for the spec's data model:
#       runs     (run_id PK, model, prompt_version, feedback_level, max_tries, timestamp)
#       tasks    (task_id PK, prompt, test_code)
#       attempts (attempt_id PK, run_id FK, task_id FK, attempt_number,
#                 code, test_results JSON, feedback_given)
#       results  (run_id FK, task_id FK, passed, tries_taken, final_code,
#                 quality_score JSON, PRIMARY KEY (run_id, task_id))
#   - Write helpers: create_run, save_task, save_attempt, save_result.
#   - Read helpers used by regression.py and the dashboard:
#       list_runs, get_results(run_id), get_attempts(run_id, task_id).
#   - Store per-test results as JSON text (json.dumps of the dataclasses).
#   - Use parameterized queries only (no f-string SQL).

import sqlite3

from agent_eval.models import Attempt, RunInfo, Task, TaskResult


def get_connection() -> sqlite3.Connection:
    raise NotImplementedError


def init_db() -> None:
    raise NotImplementedError


def create_run(run: RunInfo) -> None:
    raise NotImplementedError


def save_task(task: Task) -> None:
    raise NotImplementedError


def save_attempt(run_id: str, task_id: str, attempt: Attempt) -> None:
    raise NotImplementedError


def save_result(run_id: str, result: TaskResult) -> None:
    raise NotImplementedError


def list_runs() -> list[dict]:
    raise NotImplementedError


def get_results(run_id: str) -> list[dict]:
    raise NotImplementedError


def get_attempts(run_id: str, task_id: str) -> list[dict]:
    raise NotImplementedError
