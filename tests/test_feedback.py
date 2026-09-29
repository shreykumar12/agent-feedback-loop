"""Feedback is what the agent learns from on each retry. These tests lock in
that the right information (and only that) reaches the prompt."""

import pytest

from agent_eval.feedback import FEEDBACK_LEVELS, MAX_CASES_SHOWN, MAX_FEEDBACK_CHARS, build_feedback
from agent_eval.models import TestCaseResult, TestRunResult


def _wrong_answer_result():
    return TestRunResult(passed=False, cases=[
        TestCaseResult(name="test_ok", passed=True),
        TestCaseResult(name="test_negative", passed=False, error_type="wrong_answer",
                       expected="-2", actual="2", error="expected -2, got 2",
                       assertion="assert add(-1, -1) == -2"),
        TestCaseResult(name="test_crash", passed=False, error_type="runtime_error",
                       error="Traceback...\nZeroDivisionError: division by zero",
                       assertion="assert add(0, 0) == 0"),
    ])


def test_full_feedback_includes_failures():
    text = build_feedback(_wrong_answer_result(), "full")
    assert "1/3 tests passed" in text
    assert "test_negative" in text and "test_crash" in text
    assert "expected: -2" in text and "actual:   2" in text
    assert "assert add(-1, -1) == -2" in text
    assert "ZeroDivisionError" in text


def test_passing_cases_are_not_listed_as_failures():
    text = build_feedback(_wrong_answer_result(), "full")
    assert "test_ok" not in text


def test_names_level_has_names_but_no_values():
    text = build_feedback(_wrong_answer_result(), "names")
    assert "test_negative" in text and "test_crash" in text
    assert "expected" not in text
    assert "assert" not in text


def test_minimal_and_none_levels_reveal_nothing():
    for level in ("minimal", "none"):
        text = build_feedback(_wrong_answer_result(), level)
        assert "test_" not in text
        assert "expected" not in text


def test_timeout_feedback():
    result = TestRunResult(passed=False, timed_out=True, cases=[
        TestCaseResult(name="test_big", passed=False, error_type="timeout",
                       error="Test exceeded 2s", assertion="assert f(10**6) == 1"),
    ])
    text = build_feedback(result, "full")
    assert "infinite loop" in text
    assert "timed out" in text


def test_syntax_error_special_case():
    result = TestRunResult(passed=False, error_type="syntax_error",
                           error="SyntaxError: expected ':' (line 1)")
    text = build_feedback(result, "full")
    assert "not valid Python" in text
    assert "SyntaxError" in text


def test_import_error_mentions_stdlib_only():
    result = TestRunResult(passed=False, error_type="import_error",
                           error="ModuleNotFoundError: No module named 'numpy'")
    assert "standard library" in build_feedback(result, "full")


def test_very_long_tracebacks_are_truncated():
    huge = "x" * 50_000
    result = TestRunResult(passed=False, cases=[
        TestCaseResult(name=f"test_{i}", passed=False, error_type="runtime_error", error=huge)
        for i in range(20)
    ])
    text = build_feedback(result, "full")
    assert len(text) <= MAX_FEEDBACK_CHARS + 100
    assert "truncated" in text
    assert f"{20 - MAX_CASES_SHOWN} more failing tests" in text


def test_hidden_test_source_is_never_dumped():
    result = _wrong_answer_result()
    text = build_feedback(result, "full")
    assert "def test_" not in text  # only single assertion lines, never the test file


def test_levels_are_monotonic_in_information():
    lengths = [len(build_feedback(_wrong_answer_result(), lvl)) for lvl in FEEDBACK_LEVELS]
    assert lengths[1] <= lengths[2] <= lengths[3]


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        build_feedback(_wrong_answer_result(), "verbose")
