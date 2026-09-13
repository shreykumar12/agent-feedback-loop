# WHY THIS FILE EXISTS:
#   One place for every knob the experiment depends on. Regression detection
#   only means something if we know exactly which model, prompt version, and
#   retry cap produced a run -- scattering these across files makes runs
#   impossible to compare.
#
# WHAT IT NEEDS:
#   - Load secrets from .env (python-dotenv). Never hardcode API keys.
#   - Model settings: generator model name, judge model name, temperature
#     (use 0 for reproducibility), base URL. Gemini exposes an
#     OpenAI-compatible endpoint, so `langchain-openai` works with
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/".
#   - Loop settings: MAX_TRIES (spec says ~3-4).
#   - Sandbox settings: per-test timeout seconds, memory limit.
#   - Paths: tasks file, SQLite DB file.
#   - Optionally allow CLI/env overrides so a regression run can swap the
#     model or prompt version without editing code.

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-2.5-flash")  # TODO: confirm model name
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")  # TODO: confirm model name
TEMPERATURE = 0.0

# --- Loop ---
MAX_TRIES = int(os.getenv("MAX_TRIES", "3"))

# --- Sandbox ---
SANDBOX_TIMEOUT_SECONDS = 10
SANDBOX_MEMORY_LIMIT_MB = 256

# --- Paths ---
TASKS_PATH = PROJECT_ROOT / "tasks" / "tasks.json"
DB_PATH = PROJECT_ROOT / "data" / "agent_eval.db"
