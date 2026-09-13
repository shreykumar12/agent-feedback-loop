# WHY THIS FILE EXISTS:
#   Shared typed data structures passed between the agent, sandbox, feedback
#   builder, graph, and storage. Without one definition of "what a test result
#   looks like", every module invents its own dict shape and they drift apart.
#
# WHAT IT NEEDS:
#   Mirror the spec's data model:
#   - Task:        task_id, prompt, entry_point (function name), test_code
#   - TestCaseResult: name, passed, error message/traceback, expected, actual
#   - TestRunResult:  all TestCaseResults + overall passed + stdout/stderr +
#                     timed_out flag (the sandbox's full output)
#   - Attempt:     attempt_number, code, TestRunResult, feedback_given
#   - TaskResult:  task_id, passed, tries_taken, final_code, quality_score
#   - RunInfo:     run_id, model, prompt_version, timestamp
#   Use dataclasses or pydantic (already installed). Keep them plain data --
#   no logic here.

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    prompt: str
    entry_point: str
    test_code: str


@dataclass
class TestCaseResult:
    name: str
    passed: bool
    error: str | None = None
    expected: str | None = None
    actual: str | None = None


@dataclass
class TestRunResult:
    passed: bool
    cases: list[TestCaseResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass
class Attempt:
    attempt_number: int
    code: str
    test_result: TestRunResult
    feedback_given: str | None = None


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    tries_taken: int
    final_code: str
    quality_score: dict | None = None


@dataclass
class RunInfo:
    run_id: str
    model: str
    prompt_version: str
    timestamp: str
