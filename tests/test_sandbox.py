"""The sandbox is the source of truth for correctness: if it mis-scores code,
every downstream metric is wrong. Hand-written code strings, no LLM, no key."""

import time

import pytest

from agent_eval.models import Task
from agent_eval.sandbox import run_tests

TESTS = """
def test_positive():
    assert add(2, 3) == 5

def test_negative():
    assert add(-1, -1) == -2

def test_zero():
    assert add(0, 0) == 0
"""


@pytest.fixture
def task():
    return Task(task_id="add", prompt="def add(a, b): ...", entry_point="add", test_code=TESTS)


def test_correct_solution_passes(task):
    result = run_tests("def add(a, b):\n    return a + b\n", task)
    assert result.passed
    assert [c.name for c in result.cases] == ["test_positive", "test_negative", "test_zero"]
    assert all(c.passed for c in result.cases)
    assert result.primary_error_type is None


def test_wrong_answer_fails_with_details(task):
    result = run_tests("def add(a, b):\n    return abs(a) + abs(b)\n", task)
    assert not result.passed
    failing = result.failing
    assert [c.name for c in failing] == ["test_negative"]
    case = failing[0]
    assert case.error_type == "wrong_answer"
    assert case.expected == "-2"
    assert case.actual == "2"
    assert case.assertion == "assert add(-1, -1) == -2"
    assert result.primary_error_type == "wrong_answer"


def test_partial_pass_breakdown(task):
    result = run_tests("def add(a, b):\n    return 5\n", task)
    assert [c.passed for c in result.cases] == [True, False, False]
    assert result.num_passed == 1 and result.num_total == 3
    assert result.pass_fraction == pytest.approx(1 / 3)


def test_syntax_error_is_a_result_not_exception(task):
    result = run_tests("def add(a, b)\n    return a + b\n", task)
    assert not result.passed
    assert result.error_type == "syntax_error"
    assert "SyntaxError" in result.error
    assert result.cases == []


def test_import_error_for_non_stdlib_module(task):
    result = run_tests("import numpy\n\ndef add(a, b):\n    return a + b\n", task)
    assert result.error_type == "import_error"
    assert "numpy" in result.error
    assert "/tmp" not in result.error  # temp paths are scrubbed


def test_missing_entry_point(task):
    result = run_tests("def plus(a, b):\n    return a + b\n", task)
    assert result.error_type == "missing_entry_point"


def test_runtime_error_is_captured_per_test(task):
    result = run_tests("def add(a, b):\n    return a + b / (a + 1)\n", task)
    by_name = {c.name: c for c in result.cases}
    assert by_name["test_negative"].error_type == "runtime_error"
    assert "ZeroDivisionError" in by_name["test_negative"].error


def test_infinite_loop_times_out(task):
    code = "def add(a, b):\n    if a < 0:\n        while True:\n            pass\n    return a + b\n"
    start = time.perf_counter()
    result = run_tests(code, task, per_test_timeout=0.5)
    assert time.perf_counter() - start < 5
    assert result.timed_out
    by_name = {c.name: c for c in result.cases}
    assert by_name["test_negative"].error_type == "timeout"
    assert by_name["test_positive"].passed  # other tests still ran


def test_timeout_cannot_be_swallowed_by_except_exception(task):
    code = ("def add(a, b):\n    try:\n        while True:\n            pass\n"
            "    except Exception:\n        return a + b\n")
    result = run_tests(code, task, per_test_timeout=0.3)
    assert not result.passed
    assert all(c.error_type == "timeout" for c in result.cases)


def test_module_level_infinite_loop(task):
    result = run_tests("while True:\n    pass\n", task, per_test_timeout=0.5)
    assert result.error_type == "timeout"
    assert result.timed_out


def test_global_timeout_backstop_kills_process(task):
    # Blocks in C without returning to the interpreter; only the global timeout stops it.
    code = "import time\ndef add(a, b):\n    time.sleep(30)\n"
    start = time.perf_counter()
    result = run_tests(code, task, timeout=1.5, per_test_timeout=20)
    assert time.perf_counter() - start < 6
    assert result.timed_out
    assert not result.passed
    assert result.cases and result.cases[0].error_type == "timeout"


def test_secrets_do_not_leak_into_sandbox(task, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    code = ("import os\n\ndef add(a, b):\n"
            "    assert 'GEMINI_API_KEY' not in os.environ\n"
            "    assert 'super-secret-value' not in repr(dict(os.environ))\n"
            "    return a + b\n")
    assert run_tests(code, task).passed


def test_memory_limit(task):
    code = "def add(a, b):\n    x = bytearray(4 * 1024 ** 3)\n    return a + b\n"
    result = run_tests(code, task, memory_mb=256)
    assert not result.passed
    assert result.primary_error_type == "memory_error"


def test_sys_exit_is_a_failure_not_a_pass(task):
    result = run_tests("import sys\ndef add(a, b):\n    sys.exit(0)\n", task)
    assert not result.passed
    assert all(c.error_type == "runtime_error" for c in result.cases)


def test_stdout_is_captured(task):
    result = run_tests("def add(a, b):\n    print('debug', a)\n    return a + b\n", task)
    assert result.passed
    assert "debug 2" in result.stdout


def test_runs_are_isolated_from_each_other(task):
    writer = "open('state.txt', 'w').write('x')\ndef add(a, b):\n    return a + b\n"
    reader = "import os\ndef add(a, b):\n    assert not os.path.exists('state.txt')\n    return a + b\n"
    assert run_tests(writer, task).passed
    assert run_tests(reader, task).passed


def test_non_eq_assertions_still_report_line(task):
    t = Task(task_id="x", prompt="", entry_point="add",
             test_code="def test_gt():\n    assert add(1, 1) > 5\n")
    result = run_tests("def add(a, b):\n    return a + b\n", t)
    assert result.cases[0].error_type == "wrong_answer"
    assert result.cases[0].assertion == "assert add(1, 1) > 5"
