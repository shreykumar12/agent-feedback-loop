"""Shared typed data structures.

Every module (agent, sandbox, feedback, graph, storage, metrics) passes these
around, so there is exactly one definition of "what a test result looks like".
They are plain data -- no logic beyond (de)serialisation helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Error taxonomy used by the sandbox, feedback builder and metrics. Ordered
# roughly from "code never ran" to "code ran and was wrong".
ERROR_TYPES = (
    "syntax_error",
    "import_error",
    "missing_entry_point",
    "timeout",
    "memory_error",
    "runtime_error",
    "wrong_answer",
    "crash",
)


@dataclass
class Task:
    task_id: str
    prompt: str
    entry_point: str
    test_code: str
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    # Reference solution + plausible buggy variants. Never shown to a real
    # model: they validate the test suite (mutation score) and drive the
    # offline simulated agent.
    canonical_solution: str | None = None
    mutants: list[str] = field(default_factory=list)


@dataclass
class TestCaseResult:
    name: str
    passed: bool
    error: str | None = None
    error_type: str | None = None
    expected: str | None = None
    actual: str | None = None
    # The failing source line from the test (e.g. "assert add(2, 3) == 5").
    # This -- not the whole hidden test file -- is what "full" feedback shows.
    assertion: str | None = None
    duration_s: float = 0.0

    __test__ = False  # stop pytest from collecting this as a test class


@dataclass
class TestRunResult:
    passed: bool
    cases: list[TestCaseResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    # Set when the whole module failed before any test ran (syntax/import
    # error, missing function) or the process died.
    error_type: str | None = None
    error: str | None = None
    duration_s: float = 0.0

    __test__ = False

    @property
    def num_passed(self) -> int:
        return sum(c.passed for c in self.cases)

    @property
    def num_total(self) -> int:
        return len(self.cases)

    @property
    def pass_fraction(self) -> float:
        return self.num_passed / self.num_total if self.cases else 0.0

    @property
    def failing(self) -> list[TestCaseResult]:
        return [c for c in self.cases if not c.passed]

    @property
    def primary_error_type(self) -> str | None:
        """The single error class that best describes this run (None if passed)."""
        if self.passed:
            return None
        if self.error_type:
            return self.error_type
        failing_types = [c.error_type for c in self.failing if c.error_type]
        for kind in ERROR_TYPES:
            if kind in failing_types:
                return kind
        return "wrong_answer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestRunResult:
        data = dict(data)
        data["cases"] = [TestCaseResult(**c) for c in data.get("cases", [])]
        return cls(**data)


@dataclass
class Attempt:
    attempt_number: int
    code: str
    test_result: TestRunResult
    feedback_given: str | None = None
    # Feedback produced *from* this attempt's failure (fed into the next one).
    feedback_produced: str | None = None
    raw_response: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    tries_taken: int
    final_code: str
    quality_score: dict | None = None
    code_metrics: dict | None = None
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_s: float = 0.0
    error: str | None = None  # unexpected harness error, if the task crashed


@dataclass
class RunInfo:
    run_id: str
    model: str
    prompt_version: str
    timestamp: str
    feedback_level: str = "full"
    max_tries: int = 3
    suite_hash: str = ""
    seed: int = 0
    notes: str = ""
