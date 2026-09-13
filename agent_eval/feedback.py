# WHY THIS FILE EXISTS:
#   Engineering pillar #3: turning raw test failures into signal the agent can
#   actually act on. The quality of this feedback is the main lever on whether
#   retries help -- and measuring that effect is one of the project's headline
#   results. Keeping it separate lets you swap feedback strategies and compare.
#
# WHAT IT NEEDS:
#   - build_feedback(test_result, level="full") -> str
#   - Include (per spec): failing test names, error messages / tracebacks,
#     expected vs actual output.
#   - Special-case: timeouts ("likely infinite loop or too slow"), syntax
#     errors, and import errors -- these need different guidance than a
#     wrong answer.
#   - Truncate long tracebacks/stdout so the prompt doesn't blow up.
#   - Support multiple feedback levels, e.g.:
#       "minimal" -> "tests failed"
#       "names"   -> failing test names only
#       "full"    -> names + errors + expected/actual
#     so you can run an ablation and show richer feedback -> higher pass rate.
#   - Decide (and document) whether hidden test source is ever shown to the
#     agent. Showing full tests makes it too easy to overfit.

from agent_eval.models import TestRunResult


def build_feedback(test_result: TestRunResult, level: str = "full") -> str:
    raise NotImplementedError
