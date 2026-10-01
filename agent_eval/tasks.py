"""Load the task suite from tasks/tasks.json and validate it.

Keeping tasks as data means the suite is identical across runs -- a
requirement for fair regression comparison. ``suite_hash`` fingerprints the
suite so ``compare`` can warn when two runs were not scored on the same tests.

``validate_suite`` checks the suite itself: every canonical solution must pass
all of its tests, and every mutant (a plausible buggy variant) should be
killed by at least one test. The fraction of mutants killed is the suite's
*mutation score* -- a measure of how much the hidden tests can be trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval import config
from agent_eval.models import Task

REQUIRED_FIELDS = ("task_id", "prompt", "entry_point", "test_code")
DIFFICULTIES = ("easy", "medium", "hard")
CATEGORIES = ("function", "bugfix", "stateful", "spec", "performance")


class TaskFormatError(ValueError):
    pass


def suite_path(suite: str | None) -> Path:
    """A suite name ("easy", "hard") or a path to a tasks JSON file."""
    if not suite:
        return Path(config.TASKS_PATH) if config.DEFAULT_SUITE == "easy" else suite_path(config.DEFAULT_SUITE)
    if suite in config.SUITES:
        return Path(config.SUITES[suite])
    path = Path(suite)
    if not path.exists():
        raise TaskFormatError(f"unknown suite {suite!r}: use one of {sorted(config.SUITES)} or a path to a JSON file")
    return path


def suite_name(suite: str | None) -> str:
    if not suite:
        return config.DEFAULT_SUITE
    return suite if suite in config.SUITES else Path(suite).stem


def load_tasks(path: Path | None = None, task_ids: list[str] | None = None,
               suite: str | None = None) -> list[Task]:
    path = Path(path) if path else suite_path(suite)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TaskFormatError(f"{path} must contain a JSON list of tasks")

    tasks: list[Task] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise TaskFormatError(f"task #{i} ({entry.get('task_id', '?')}) is missing {missing}")
        if entry["task_id"] in seen:
            raise TaskFormatError(f"duplicate task_id {entry['task_id']!r}")
        seen.add(entry["task_id"])
        difficulty = entry.get("difficulty", "medium")
        if difficulty not in DIFFICULTIES:
            raise TaskFormatError(f"{entry['task_id']}: difficulty must be one of {DIFFICULTIES}")
        category = entry.get("category", "function")
        if category not in CATEGORIES:
            raise TaskFormatError(f"{entry['task_id']}: category must be one of {CATEGORIES}")
        tasks.append(Task(
            task_id=entry["task_id"],
            prompt=entry["prompt"],
            entry_point=entry["entry_point"],
            test_code=entry["test_code"],
            difficulty=difficulty,
            tags=list(entry.get("tags", [])),
            canonical_solution=entry.get("canonical_solution"),
            mutants=list(entry.get("mutants", [])),
            category=category,
            per_test_timeout=entry.get("per_test_timeout"),
        ))

    if task_ids:
        wanted = set(task_ids)
        unknown = wanted - seen
        if unknown:
            raise TaskFormatError(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t.task_id in wanted]
    return tasks


def suite_hash(tasks: list[Task]) -> str:
    """Fingerprint of what is being measured (prompts + tests), order-independent."""
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda t: t.task_id):
        for part in (task.task_id, task.prompt, task.entry_point, task.test_code):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()[:12]


@dataclass
class TaskValidation:
    task_id: str
    num_tests: int
    canonical_passed: bool | None
    canonical_failures: list[str] = field(default_factory=list)
    mutants_total: int = 0
    mutants_killed: int = 0
    surviving_mutants: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.canonical_passed is not False and not self.surviving_mutants


def validate_task(task: Task) -> TaskValidation:
    from agent_eval.sandbox import run_tests

    report = TaskValidation(task_id=task.task_id, num_tests=0, canonical_passed=None)
    if task.canonical_solution:
        result = run_tests(task.canonical_solution, task)
        report.num_tests = result.num_total
        report.canonical_passed = result.passed
        report.canonical_failures = [
            f"{c.name}: {c.error}" for c in result.failing
        ] or ([result.error] if result.error else [])
    for i, mutant in enumerate(task.mutants):
        report.mutants_total += 1
        if run_tests(mutant, task).passed:
            report.surviving_mutants.append(i)
        else:
            report.mutants_killed += 1
    return report


def validate_suite(tasks: list[Task]) -> list[TaskValidation]:
    return [validate_task(t) for t in tasks]
