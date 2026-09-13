# WHY THIS FILE EXISTS:
#   Build step 6 (optional, secondary): an LLM judge for what tests CAN'T
#   measure -- readability and approach. It is deliberately NOT used for
#   correctness; that's the sandbox's job. Keeping it in its own module makes
#   that separation obvious and lets the whole layer be switched off.
#
# WHAT IT NEEDS:
#   - A structured rubric to reduce LLM-judge inconsistency, e.g.:
#       readability: 1-5, approach: 1-5, short justification for each
#   - Force structured output (pydantic model + `with_structured_output`) so
#     scores are parseable, not free text.
#   - score_quality(task, final_code) -> dict
#   - Temperature 0; include the rubric definitions for each score level in
#     the prompt.
#   - Only run on passing code (judging broken code's "readability" is noise).
#   - Return None / skip gracefully on API errors -- quality is optional.

from agent_eval.models import Task


def score_quality(task: Task, final_code: str) -> dict | None:
    raise NotImplementedError
