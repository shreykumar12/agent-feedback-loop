# WHY THIS FILE EXISTS:
#   The coding agent itself -- the "commodity" component the spec says the
#   loop is built around. Isolating it here means the rest of the system
#   doesn't care which model or provider is used, which is what makes
#   model-swap regression comparisons possible.
#
# WHAT IT NEEDS:
#   - Build a chat model client from config (ChatOpenAI pointed at the Gemini
#     OpenAI-compatible base URL, temperature 0).
#   - generate_code(task, previous_code=None, feedback=None) -> str
#       * First attempt: use FIRST_ATTEMPT_TEMPLATE.
#       * Retry: use RETRY_TEMPLATE with previous code + feedback.
#   - extract_code(response_text) -> str
#       * Pull the code out of a ```python fenced block; fall back to raw text.
#       * Malformed output should NOT crash the loop -- return what you have
#         and let the sandbox fail it, so it counts as a real failed attempt.
#   - Optionally record token usage per call for cost tracking.

from agent_eval.models import Task


def generate_code(task: Task, previous_code: str | None = None, feedback: str | None = None) -> str:
    raise NotImplementedError


def extract_code(response_text: str) -> str:
    raise NotImplementedError
