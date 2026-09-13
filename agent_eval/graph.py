# WHY THIS FILE EXISTS:
#   Engineering pillar #4: the bounded retry loop. LangGraph models the
#   generate -> test -> (pass? done : feedback -> generate) cycle as an explicit
#   graph with a conditional edge, which makes the stopping condition visible
#   and testable instead of hidden inside a while-loop.
#
# WHAT IT NEEDS:
#   - A state TypedDict, e.g.:
#       task, attempt_number, current_code, test_result, feedback,
#       attempts (list[Attempt]), passed
#   - Nodes:
#       generate(state)  -> calls agent.generate_code, increments attempt_number
#       test(state)      -> calls sandbox.run_tests, appends an Attempt
#       feedback(state)  -> calls feedback.build_feedback
#   - Conditional edge after `test`:
#       passed                      -> END
#       attempt_number >= MAX_TRIES -> END (record as failed)
#       otherwise                   -> feedback -> generate
#   - build_graph() -> compiled graph
#   - run_task(task) -> (TaskResult, list[Attempt])
#     Converts final graph state into storable results. Should never raise on
#     a bad model response -- a failure is data.

from agent_eval.models import Attempt, Task, TaskResult


def build_graph():
    raise NotImplementedError


def run_task(task: Task) -> tuple[TaskResult, list[Attempt]]:
    raise NotImplementedError
