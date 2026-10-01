"""Optional LLM judge for what tests CAN'T measure: readability and approach.

Deliberately NOT used for correctness -- that is the sandbox's job. It only
scores code that already passed, uses a fixed rubric with anchored score
levels, forces structured output, runs at temperature 0, and returns None on
any error: quality is a nice-to-have, never a reason for a run to fail.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_eval import config
from agent_eval.models import Task

RUBRIC = """Score the solution on two axes, each an integer 1-5.

READABILITY
5 - idiomatic, clearly named, easy to follow at a glance; comments only where useful
4 - clear with minor issues (a vague name, slightly long function)
3 - understandable but needs effort (dense logic, poor names, some duplication)
2 - hard to follow (deep nesting, magic numbers, confusing control flow)
1 - nearly unreadable

APPROACH
5 - an appropriate algorithm with good complexity; handles edge cases cleanly
4 - sound approach with a small inefficiency or awkward edge-case handling
3 - works but is clearly suboptimal (needless quadratic work, special-casing)
2 - brittle or convoluted approach that happens to pass
1 - hard-coded / test-gaming or otherwise inappropriate

Correctness is already verified by tests; do NOT judge correctness."""


class QualityScore(BaseModel):
    readability: int = Field(ge=1, le=5)
    readability_reason: str
    approach: int = Field(ge=1, le=5)
    approach_reason: str


def score_quality(task: Task, final_code: str, model: str | None = None) -> dict | None:
    model = model or config.JUDGE_MODEL
    if model.startswith(("sim", "hf:")) or not config.LLM_API_KEY:
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                         temperature=0, timeout=config.LLM_TIMEOUT_SECONDS)
        structured = llm.with_structured_output(QualityScore)
        score = structured.invoke([
            SystemMessage(content="You are a strict senior Python code reviewer.\n\n" + RUBRIC),
            HumanMessage(content=f"Task specification:\n```python\n{task.prompt}\n```\n\n"
                                 f"Solution (passes all tests):\n```python\n{final_code}\n```"),
        ])
        return score.model_dump() | {"judge_model": model}
    except Exception:
        return None
