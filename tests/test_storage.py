"""Regression detection and the dashboard read from SQLite, so writes must
round-trip. Uses a temp DB so tests never touch real run history."""

import pytest

from agent_eval import storage
from agent_eval.models import Attempt, RunInfo, TaskResult, TestCaseResult, TestRunResult
from tests.helpers import make_task, store_run


def test_init_db_idempotent(tmp_db):
    storage.init_db()
    storage.init_db()
    assert storage.list_runs() == []


def test_result_round_trip(tmp_db):
    run = RunInfo(run_id="r1", model="gemini-2.5-flash", prompt_version="v2",
                  timestamp="2026-01-01T00:00:00", feedback_level="names", max_tries=4,
                  suite_hash="h", seed=7, notes="hello")
    storage.create_run(run)
    task = make_task("t1", "hard")
    tr = TestRunResult(passed=False, stdout="out", cases=[
        TestCaseResult(name="test_x", passed=False, error_type="wrong_answer", expected="1",
                       actual="2", assertion="assert f() == 1"),
        TestCaseResult(name="test_y", passed=True),
    ])
    attempt = Attempt(attempt_number=1, code="def f(): return 2", test_result=tr,
                      feedback_produced="fb", raw_response="raw", tokens_in=11, tokens_out=7, latency_s=0.5)
    result = TaskResult(task_id="t1", passed=False, tries_taken=1, final_code="def f(): return 2",
                        quality_score={"readability": 4}, code_metrics={"sloc": 1})
    storage.save_task_run("r1", task, result, [attempt])

    stored_run = storage.get_run("r1")
    assert stored_run["feedback_level"] == "names" and stored_run["seed"] == 7
    assert stored_run["status"] == "running"
    storage.finish_run("r1", "2026-01-01T00:01:00")
    assert storage.get_run("r1")["status"] == "completed"

    [res] = storage.get_results("r1")
    assert res["passed"] is False and res["tries_taken"] == 1
    assert res["quality_score"] == {"readability": 4}
    assert res["code_metrics"] == {"sloc": 1}
    assert res["difficulty"] == "hard"

    [att] = storage.get_attempts("r1", "t1")
    assert att["error_type"] == "wrong_answer"
    assert att["num_passed"] == 1 and att["num_total"] == 2
    assert att["tokens_in"] == 11 and att["feedback_produced"] == "fb"
    # JSON test results survive the round trip and rebuild into dataclasses
    rebuilt = TestRunResult.from_dict(att["test_results"])
    assert rebuilt == tr


def test_list_runs_aggregates_pass_rate(tmp_db):
    store_run("run-a", {"t1": [True], "t2": [False, False, False]})
    [run] = storage.list_runs()
    assert run["num_tasks"] == 2 and run["num_passed"] == 1
    assert run["pass_rate"] == 0.5


def test_resolve_run_id(tmp_db):
    store_run("aaaa1111", {"t1": [True]}, seed=1)
    store_run("bbbb2222", {"t1": [True]}, seed=2)
    assert storage.resolve_run_id("aaaa") == "aaaa1111"
    assert storage.resolve_run_id("latest") == "bbbb2222"
    assert storage.resolve_run_id("latest~1") == "aaaa1111"
    with pytest.raises(KeyError):
        storage.resolve_run_id("zzz")


def test_delete_run(tmp_db):
    store_run("gone", {"t1": [False, True]})
    storage.delete_run("gone")
    assert storage.get_run("gone") is None
    assert storage.get_attempts("gone") == []


def test_task_upsert(tmp_db):
    storage.save_task(make_task("t1", "easy"))
    storage.save_task(make_task("t1", "hard"))
    assert storage.get_task("t1")["difficulty"] == "hard"
