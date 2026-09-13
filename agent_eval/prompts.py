# WHY THIS FILE EXISTS:
#   Prompts are an experimental variable. The spec's regression detection
#   compares runs across (model, prompt_version), so prompts must live in one
#   versioned place rather than inline strings buried in agent code.
#
# WHAT IT NEEDS:
#   - A PROMPT_VERSION identifier recorded with every run.
#   - A system prompt telling the model to return ONLY a Python function
#     implementation (no prose, or inside a single fenced code block that the
#     agent can reliably extract).
#   - A first-attempt template: task prompt -> code.
#   - A retry template: task prompt + previous code + structured feedback -> fixed code.
#   - Optionally a registry {version: templates} so you can run v1 vs v2 and
#     measure the effect (e.g. "richer feedback wording improved pass rate").

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """TODO: instruct the model to act as a Python coding agent and
return only a single fenced python code block containing the solution."""

FIRST_ATTEMPT_TEMPLATE = """TODO: include {task_prompt}"""

RETRY_TEMPLATE = """TODO: include {task_prompt}, {previous_code}, and {feedback}"""
