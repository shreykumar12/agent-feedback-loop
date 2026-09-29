"""Builders for hand-made runs, shared by storage/metrics/regression tests."""

from agent_eval import storage
from agent_eval.models import Attempt, RunInfo, Task, TaskResult, TestCaseResult, TestRunResult


def make_task(task_id, difficulty="medium"):
    return Task(task_id=task_id, prompt=f"def {task_id}(): ...", entry_point=task_id,
                test_code="def test_a():\n    assert True\n", difficulty=difficulty)


def make_test_result(passed, n_pass=None, n_total=2, error_type="wrong_answer"):
    if passed:
        n_pass = n_total
    n_pass = n_total - 1 if n_pass is None else n_pass
    cases = [TestCaseResult(name=f"test_{i}", passed=i < n_pass,
                            error_type=None if i < n_pass else error_type) for i in range(n_total)]
    return TestRunResult(passed=passed, cases=cases)


def store_run(run_id, outcomes, model="m", prompt_version="v1", max_tries=3, seed=0,
              feedback_level="full", suite_hash="abc"):
    """outcomes: {task_id: [attempt spec, ...]} where a spec is True (pass),
    False (wrong answer), or an error_type string."""
    storage.create_run(RunInfo(run_id=run_id, model=model, prompt_version=prompt_version,
                               timestamp=f"2026-01-01T00:00:0{seed}", feedback_level=feedback_level,
                               max_tries=max_tries, suite_hash=suite_hash, seed=seed))
    for task_id, specs in outcomes.items():
        difficulty = "hard" if task_id.startswith("h") else "easy"
        task = make_task(task_id, difficulty)
        attempts = []
        for i, spec in enumerate(specs, 1):
            if spec is True:
                tr = make_test_result(True)
            elif spec is False:
                tr = make_test_result(False)
            else:
                tr = TestRunResult(passed=False, error_type=spec, error=spec)
            attempts.append(Attempt(attempt_number=i, code=f"# {task_id} v{i}\n", test_result=tr,
                                    tokens_in=100, tokens_out=50, latency_s=1.0))
        passed = bool(specs) and specs[-1] is True
        result = TaskResult(task_id=task_id, passed=passed, tries_taken=len(specs),
                            final_code=attempts[-1].code, total_tokens_in=100 * len(specs),
                            total_tokens_out=50 * len(specs), total_latency_s=float(len(specs)))
        storage.save_task_run(run_id, task, result, attempts)
