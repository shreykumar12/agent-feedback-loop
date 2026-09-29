"""Versioned prompts. Prompts are an experimental variable: every run records
its prompt_version, and regression comparisons are made across versions.

v1 -- minimal instructions.
v2 -- asks the model to reason about edge cases stated in the docstring before
      writing code, and on retries to diagnose the failure and change as little
      as possible (targeted repair instead of a rewrite).
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "v1"  # default


@dataclass(frozen=True)
class PromptSet:
    system: str
    first_attempt: str
    retry: str


_V1 = PromptSet(
    system=(
        "You are a Python coding agent. You are given a function signature and docstring. "
        "Implement the function using only the Python standard library. "
        "Respond with a single fenced ```python code block containing the complete "
        "implementation (imports, helpers, and the function). No explanations."
    ),
    first_attempt=(
        "Implement the following function:\n\n```python\n{task_prompt}\n```"
    ),
    retry=(
        "Implement the following function:\n\n```python\n{task_prompt}\n```\n\n"
        "Your previous attempt:\n\n```python\n{previous_code}\n```\n\n"
        "It was run against hidden tests. Test feedback:\n\n{feedback}\n\n"
        "Return the corrected complete implementation in a single ```python code block."
    ),
)

_V2 = PromptSet(
    system=(
        "You are a meticulous Python engineer. Implement functions exactly as their docstrings "
        "specify, using only the Python standard library. The docstring is the full specification: "
        "every stated edge case, tie-breaking rule, rounding rule and error condition is tested. "
        "Before coding, silently list the edge cases the docstring mentions and make sure the code "
        "handles each. Respond with ONLY a single fenced ```python code block containing the complete "
        "implementation. No prose outside the code block."
    ),
    first_attempt=(
        "Task:\n\n```python\n{task_prompt}\n```\n\n"
        "Pay special attention to edge cases and conventions stated in the docstring."
    ),
    retry=(
        "Task:\n\n```python\n{task_prompt}\n```\n\n"
        "Your previous attempt:\n\n```python\n{previous_code}\n```\n\n"
        "Hidden test feedback:\n\n{feedback}\n\n"
        "Diagnose the root cause of each failure (compare expected vs actual against the docstring), "
        "then fix it with the smallest change that makes the code correct for ALL cases in the "
        "docstring, not just the failing ones. Return the complete corrected implementation in a "
        "single ```python code block."
    ),
)

PROMPTS: dict[str, PromptSet] = {"v1": _V1, "v2": _V2}

# Back-compat names for the default version.
SYSTEM_PROMPT = _V1.system
FIRST_ATTEMPT_TEMPLATE = _V1.first_attempt
RETRY_TEMPLATE = _V1.retry


def get_prompts(version: str) -> PromptSet:
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(f"unknown prompt version {version!r}; available: {sorted(PROMPTS)}") from None


def render(version: str, task_prompt: str, previous_code: str | None = None,
           feedback: str | None = None) -> tuple[str, str]:
    """Return (system, user) messages for a first attempt or a retry."""
    prompts = get_prompts(version)
    if previous_code is None:
        return prompts.system, prompts.first_attempt.format(task_prompt=task_prompt)
    return prompts.system, prompts.retry.format(
        task_prompt=task_prompt, previous_code=previous_code, feedback=feedback or "(no feedback)"
    )
