"""Turn raw test results into signal the agent can act on.

Feedback quality is the main lever on whether retries help, so it is
parameterised by *level* and ablated as an experiment:

    none     -> nothing beyond "try again" (pure resampling baseline)
    minimal  -> "tests failed"
    names    -> pass count + failing test names
    full     -> names + failing assertion line + expected/actual + error/traceback,
                with special-case guidance for timeouts, syntax, import errors

Hidden-test policy: the agent never sees the test file. "full" shows only the
single failing assertion line for each failing test (what a developer sees in
a pytest report), capped at MAX_CASES_SHOWN failures. This keeps feedback
actionable while making it hard to overfit to the whole suite.
"""

from __future__ import annotations

from agent_eval.models import TestCaseResult, TestRunResult

FEEDBACK_LEVELS = ("none", "minimal", "names", "full")
MAX_CASES_SHOWN = 5
MAX_ERROR_CHARS = 600
MAX_FEEDBACK_CHARS = 3500

_MODULE_GUIDANCE = {
    "syntax_error": "Your code is not valid Python, so no tests could run. Fix the syntax error below.",
    "import_error": (
        "Your code failed to import. Only the Python standard library is available -- "
        "remove any third-party imports (numpy, pandas, ...) and check module names."
    ),
    "missing_entry_point": "The required function is not defined. Keep the exact function name and signature from the task.",
    "timeout": (
        "Your code timed out while being imported. Module-level code must not run forever; "
        "only define functions at the top level."
    ),
    "memory_error": "Your code exceeded the memory limit while being imported.",
    "runtime_error": "Your code raised an exception at import time (module-level code), so no tests could run.",
    "crash": "The test process crashed before producing results.",
}

_CASE_GUIDANCE = {
    "timeout": "likely an infinite loop or an algorithm that is too slow for this input",
    "memory_error": "the code allocates far too much memory for this input",
    "runtime_error": "the code raised an exception instead of returning a value",
}


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n  ...<{len(text) - limit} chars truncated>...\n{text[-tail:]}"


def _describe_case(case: TestCaseResult) -> str:
    lines = [f"- {case.name}"]
    if case.assertion:
        lines.append(f"    test line: {_truncate(case.assertion, 300)}")
    if case.expected is not None or case.actual is not None:
        lines.append(f"    expected: {case.expected}")
        lines.append(f"    actual:   {case.actual}")
    elif case.error:
        error = _truncate(case.error, MAX_ERROR_CHARS).replace("\n", "\n      ")
        lines.append(f"    error: {error}")
    hint = _CASE_GUIDANCE.get(case.error_type or "")
    if hint:
        lines.append(f"    note: {hint}")
    return "\n".join(lines)


def build_feedback(test_result: TestRunResult, level: str = "full") -> str:
    if level not in FEEDBACK_LEVELS:
        raise ValueError(f"unknown feedback level {level!r}; expected one of {FEEDBACK_LEVELS}")
    if test_result.passed:
        return "All tests passed."
    if level == "none":
        return "Your previous solution was not accepted. Try again."
    if level == "minimal":
        return "Your previous solution failed the hidden tests."

    # Module-level failure: no per-test results exist.
    if not test_result.cases:
        kind = test_result.error_type or "crash"
        if level == "names":
            return f"Your previous solution failed before any test could run ({kind.replace('_', ' ')})."
        parts = [_MODULE_GUIDANCE.get(kind, _MODULE_GUIDANCE["crash"])]
        if test_result.error:
            parts.append(_truncate(test_result.error, MAX_ERROR_CHARS * 2))
        return _truncate("\n\n".join(parts), MAX_FEEDBACK_CHARS)

    failing = test_result.failing
    summary = f"{test_result.num_passed}/{test_result.num_total} tests passed."
    if level == "names":
        names = ", ".join(c.name for c in failing)
        return f"{summary} Failing tests: {names}"

    parts = [f"{summary} Failing tests:"]
    parts += [_describe_case(c) for c in failing[:MAX_CASES_SHOWN]]
    if len(failing) > MAX_CASES_SHOWN:
        parts.append(f"...and {len(failing) - MAX_CASES_SHOWN} more failing tests not shown.")
    if test_result.timed_out:
        parts.append("At least one test timed out: check loop termination conditions and algorithmic complexity.")
    return _truncate("\n".join(parts), MAX_FEEDBACK_CHARS)
