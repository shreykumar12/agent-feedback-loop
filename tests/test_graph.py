"""The loop's control flow -- the stopping conditions are the core engineering
claim. Uses a FAKE agent: deterministic, free, runs in CI without a key."""

import pytest

from agent_eval import agent
from agent_eval.agent import Generation
from agent_eval.graph import LoopSettings, route_after_test, run_task
from agent_eval.models import Task

CORRECT = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"
BROKEN = "def add(a, b)\n    return a + b\n"


@pytest.fixture
def task():
    return Task(task_id="add", prompt="def add(a, b):\n    \"\"\"Sum.\"\"\"\n", entry_point="add",
                test_code="def test_sum():\n    assert add(2, 3) == 5\n\ndef test_zero():\n    assert add(0, 0) == 0\n")


@pytest.fixture
def scripted(monkeypatch):
    """Replace the LLM with a scripted sequence of programs; record every call."""
    calls = []

    def install(programs):
        def fake_generate(task, previous_code=None, feedback=None, **kwargs):
            calls.append({"previous_code": previous_code, "feedback": feedback, **kwargs})
            code = programs[min(len(calls), len(programs)) - 1]
            return Generation(code=code, raw=f"```python\n{code}```", tokens_in=10, tokens_out=5,
                              latency_s=0.1)
        monkeypatch.setattr(agent, "generate", fake_generate)
        return calls

    return install


def test_passes_first_try(task, scripted):
    calls = scripted([CORRECT])
    result, attempts = run_task(task, LoopSettings(model="fake", max_tries=3))
    assert result.passed and result.tries_taken == 1
    assert len(calls) == 1
    assert calls[0]["previous_code"] is None and calls[0]["feedback"] is None
    assert attempts[0].feedback_given is None


def test_passes_after_retries(task, scripted):
    scripted([BROKEN, WRONG, CORRECT])
    result, attempts = run_task(task, LoopSettings(model="fake", max_tries=4))
    assert result.passed and result.tries_taken == 3
    assert [a.test_result.primary_error_type for a in attempts] == ["syntax_error", "wrong_answer", None]
    assert result.final_code == CORRECT
    assert result.total_tokens_in == 30 and result.total_tokens_out == 15


def test_retry_prompt_receives_previous_code_and_feedback(task, scripted):
    calls = scripted([WRONG, CORRECT])
    _, attempts = run_task(task, LoopSettings(model="fake", max_tries=3, feedback_level="full"))
    retry = calls[1]
    assert retry["previous_code"] == WRONG
    assert "test_sum" in retry["feedback"] and "expected: 5" in retry["feedback"]
    assert retry["history"] == [WRONG]
    assert retry["attempt_number"] == 2
    assert attempts[1].feedback_given == retry["feedback"]
    assert attempts[0].feedback_produced == retry["feedback"]


def test_stops_at_max_tries(task, scripted):
    calls = scripted([WRONG])
    result, attempts = run_task(task, LoopSettings(model="fake", max_tries=3))
    assert not result.passed
    assert result.tries_taken == 3 and len(calls) == 3
    assert [a.attempt_number for a in attempts] == [1, 2, 3]


def test_max_tries_one_means_no_retries(task, scripted):
    calls = scripted([WRONG, CORRECT])
    result, _ = run_task(task, LoopSettings(model="fake", max_tries=1))
    assert not result.passed and len(calls) == 1


def test_feedback_level_is_respected(task, scripted):
    calls = scripted([WRONG, CORRECT])
    run_task(task, LoopSettings(model="fake", max_tries=2, feedback_level="minimal"))
    assert "test_sum" not in calls[1]["feedback"]


def test_route_after_test():
    assert route_after_test({"passed": True, "attempt_number": 1}, 3) == "done"
    assert route_after_test({"passed": False, "attempt_number": 3}, 3) == "done"
    assert route_after_test({"passed": False, "attempt_number": 2}, 3) == "retry"


def test_agent_infra_errors_propagate(task, monkeypatch):
    def boom(*args, **kwargs):
        raise agent.AgentError("quota exceeded")
    monkeypatch.setattr(agent, "generate", boom)
    with pytest.raises(agent.AgentError):
        run_task(task, LoopSettings(model="fake"))
