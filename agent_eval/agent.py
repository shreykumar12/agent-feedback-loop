"""The coding agent -- the "commodity" component the loop is built around.

The rest of the system only sees ``generate()``, so swapping the model or
provider is a config change, which is what makes model-swap regression
comparisons possible. Providers:

  * any OpenAI-compatible chat endpoint via langchain-openai (Gemini by
    default: ``gemini-3.8-flash`` etc.)
  * ``sim-*`` models: an offline, deterministic simulated agent (see
    ``simulated.py``) so the full pipeline runs without an API key.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

from agent_eval import config, prompts
from agent_eval.models import Task


class AgentError(RuntimeError):
    """The model could not be called (auth, network, quota). Not a code failure."""


def hint_for(exc: Exception) -> str:
    """A one-line suggestion for common provider errors."""
    text = str(exc)
    if "NotFound" in text or "404" in text:
        return ("\nhint: this model name is not available to your API key. Pass another with --model "
                "(or set GENERATOR_MODEL in .env); see https://ai.google.dev/gemini-api/docs/models")
    if "401" in text or "403" in text or "Authentication" in text or "PermissionDenied" in text:
        return "\nhint: the API key was rejected; check GEMINI_API_KEY (or LLM_API_KEY) in .env"
    if "429" in text or "RateLimit" in text or "quota" in text.lower():
        return "\nhint: rate limit or quota hit; retry later or lower --workers"
    return ""


@dataclass
class Generation:
    code: str
    raw: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(response_text: str, entry_point: str | None = None) -> str:
    """Pull code out of a fenced block; fall back to the raw text.

    Malformed output never raises: whatever comes back is handed to the
    sandbox, which scores it as a real (failed) attempt.
    """
    text = response_text or ""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    if not blocks:
        return text.strip() + "\n" if text.strip() else ""
    python_blocks = [b for lang, b in blocks if lang in ("python", "py", "python3")] or [b for _, b in blocks]
    if entry_point:
        defining = [b for b in python_blocks if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b)]
        if defining:
            python_blocks = defining
    return max(python_blocks, key=len).strip() + "\n"


def is_simulated(model: str) -> bool:
    return model.startswith("sim")


@lru_cache(maxsize=8)
def _chat_client(model: str):
    from langchain_openai import ChatOpenAI

    if not config.LLM_API_KEY:
        raise AgentError(
            "No API key configured. Set GEMINI_API_KEY (or LLM_API_KEY) in .env, "
            "or use an offline simulated model such as --model sim-base."
        )
    return ChatOpenAI(
        model=model,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.TEMPERATURE,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
    )


def _call_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    from langchain_core.messages import HumanMessage, SystemMessage

    client = _chat_client(model)
    try:
        message = client.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # network/auth/quota -- infrastructure, not the agent's code
        raise AgentError(f"{type(exc).__name__}: {exc}") from exc
    usage = getattr(message, "usage_metadata", None) or {}
    content = message.content
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def generate(
    task: Task,
    previous_code: str | None = None,
    feedback: str | None = None,
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    history: list[str] | None = None,
    attempt_number: int = 1,
    seed: int = 0,
) -> Generation:
    model = model or config.GENERATOR_MODEL
    prompt_version = prompt_version or config.DEFAULT_PROMPT_VERSION
    system, user = prompts.render(prompt_version, task.prompt, previous_code, feedback)

    start = time.perf_counter()
    if is_simulated(model):
        from agent_eval import simulated

        raw, tokens_in, tokens_out, latency = simulated.respond(
            model, task, system, user, feedback=feedback, history=history or [],
            attempt_number=attempt_number, seed=seed,
        )
    else:
        raw, tokens_in, tokens_out = _call_llm(model, system, user)
        latency = time.perf_counter() - start
    return Generation(
        code=extract_code(raw, task.entry_point),
        raw=raw,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_s=round(latency, 4),
    )


def generate_code(task: Task, previous_code: str | None = None, feedback: str | None = None) -> str:
    """Convenience wrapper returning only the code (default model + prompt)."""
    return generate(task, previous_code, feedback).code
