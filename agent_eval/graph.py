"""The bounded retry loop as an explicit LangGraph state machine.

    START -> generate -> test --passed--------------------> END
                          |----attempt_number >= max_tries-> END (failed)
                          '----otherwise-> feedback -> generate

Making the stopping condition a conditional edge (``route_after_test``) keeps
it visible and unit-testable instead of buried in a while-loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_eval import agent, config, sandbox
from agent_eval.code_metrics import analyze
from agent_eval.feedback import build_feedback
from agent_eval.models import Attempt, Task, TaskResult, TestRunResult


@dataclass(frozen=True)
class LoopSettings:
    model: str = config.GENERATOR_MODEL
    prompt_version: str = config.DEFAULT_PROMPT_VERSION
    feedback_level: str = config.FEEDBACK_LEVEL
    max_tries: int = config.MAX_TRIES
    seed: int = 0


class LoopState(TypedDict, total=False):
    task: Task
    attempt_number: int
    current_code: str
    generation: Any  # agent.Generation of the attempt in flight
    test_result: TestRunResult | None
    feedback: str | None
    attempts: list[Attempt]
    passed: bool


def route_after_test(state: LoopState, max_tries: int) -> str:
    if state["passed"]:
        return "done"
    if state["attempt_number"] >= max_tries:
        return "done"
    return "retry"


def build_graph(settings: LoopSettings | None = None):
    settings = settings or LoopSettings()

    def generate(state: LoopState) -> LoopState:
        attempts = state.get("attempts", [])
        attempt_number = state.get("attempt_number", 0) + 1
        generation = agent.generate(
            state["task"],
            previous_code=state.get("current_code") if attempts else None,
            feedback=state.get("feedback") if attempts else None,
            model=settings.model,
            prompt_version=settings.prompt_version,
            history=[a.code for a in attempts],
            attempt_number=attempt_number,
            seed=settings.seed,
        )
        return {"attempt_number": attempt_number, "current_code": generation.code,
                "generation": generation}

    def test(state: LoopState) -> LoopState:
        result = sandbox.run_tests(state["current_code"], state["task"])
        gen = state["generation"]
        attempt = Attempt(
            attempt_number=state["attempt_number"],
            code=state["current_code"],
            test_result=result,
            feedback_given=state.get("feedback") if state.get("attempts") else None,
            raw_response=gen.raw,
            tokens_in=gen.tokens_in,
            tokens_out=gen.tokens_out,
            latency_s=gen.latency_s,
        )
        return {"test_result": result, "passed": result.passed,
                "attempts": [*state.get("attempts", []), attempt]}

    def feedback(state: LoopState) -> LoopState:
        text = build_feedback(state["test_result"], level=settings.feedback_level)
        state["attempts"][-1].feedback_produced = text
        return {"feedback": text}

    graph = StateGraph(LoopState)
    graph.add_node("generate", generate)
    graph.add_node("test", test)
    graph.add_node("feedback", feedback)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "test")
    graph.add_conditional_edges(
        "test",
        lambda s: route_after_test(s, settings.max_tries),
        {"done": END, "retry": "feedback"},
    )
    graph.add_edge("feedback", "generate")
    return graph.compile()


def run_task(task: Task, settings: LoopSettings | None = None) -> tuple[TaskResult, list[Attempt]]:
    """Run the loop on one task. Code failures are data, never exceptions;
    only infrastructure errors (e.g. ``agent.AgentError``) propagate."""
    settings = settings or LoopSettings()
    app = build_graph(settings)
    final = app.invoke(
        {"task": task, "attempt_number": 0, "attempts": [], "feedback": None, "passed": False},
        config={"recursion_limit": 3 * settings.max_tries + 10},
    )
    attempts: list[Attempt] = final["attempts"]
    last = attempts[-1]
    result = TaskResult(
        task_id=task.task_id,
        passed=bool(final["passed"]),
        tries_taken=len(attempts),
        final_code=last.code,
        code_metrics=analyze(last.code),
        total_tokens_in=sum(a.tokens_in for a in attempts),
        total_tokens_out=sum(a.tokens_out for a in attempts),
        total_latency_s=round(sum(a.latency_s + a.test_result.duration_s for a in attempts), 4),
    )
    return result, attempts
