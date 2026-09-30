"""The suite itself must be trustworthy: every reference solution passes its
hidden tests, and every plausible buggy variant is caught (mutation score)."""

import json

import pytest

from agent_eval import config
from agent_eval.tasks import TaskFormatError, load_tasks, suite_hash, validate_task

TASKS = load_tasks()
HARD = load_tasks(suite="hard")
ALL = TASKS + HARD


def test_suite_shape():
    assert 15 <= len(TASKS) <= 25
    assert {t.difficulty for t in TASKS} == {"easy", "medium", "hard"}
    assert len({t.task_id for t in TASKS}) == len(TASKS)


def test_hard_suite_shape():
    assert 15 <= len(HARD) <= 25
    assert {t.category for t in HARD} == {"bugfix", "stateful", "spec", "performance"}
    assert sum(t.difficulty == "hard" for t in HARD) >= len(HARD) * 0.7
    # Task ids are global in storage, so the suites must not share any.
    assert len({t.task_id for t in ALL}) == len(ALL)
    for task in HARD:
        if task.category == "performance":
            assert task.per_test_timeout, f"{task.task_id} needs a per_test_timeout"


@pytest.mark.parametrize("task", ALL, ids=[t.task_id for t in ALL])
def test_canonical_passes_and_all_mutants_killed(task):
    report = validate_task(task)
    assert report.canonical_passed, report.canonical_failures
    assert report.num_tests >= 5
    assert not report.surviving_mutants, f"mutants {report.surviving_mutants} survive the tests"


def test_prompts_never_contain_tests_or_solutions():
    for task in ALL:
        assert "def test_" not in task.prompt
        assert task.canonical_solution.strip() != task.prompt.strip()


def test_filter_and_unknown_ids():
    subset = load_tasks(task_ids=[TASKS[0].task_id])
    assert [t.task_id for t in subset] == [TASKS[0].task_id]
    with pytest.raises(TaskFormatError):
        load_tasks(task_ids=["does_not_exist"])


def test_suite_hash_is_order_independent_and_sensitive(tmp_path):
    assert suite_hash(TASKS) == suite_hash(list(reversed(TASKS)))
    raw = json.loads(config.TASKS_PATH.read_text())
    raw[0]["test_code"] += "\n# changed\n"
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(raw))
    assert suite_hash(load_tasks(path)) != suite_hash(TASKS)


def test_missing_fields_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"task_id": "x", "prompt": "p"}]))
    with pytest.raises(TaskFormatError):
        load_tasks(path)
