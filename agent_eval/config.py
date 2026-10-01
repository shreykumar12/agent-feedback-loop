"""Every knob the experiment depends on, in one place.

Regression detection only means something if we know exactly which model,
prompt version, feedback level and retry cap produced a run, so all of them
live here (overridable via env vars / CLI flags) and are recorded per run.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM ---
# Gemini exposes an OpenAI-compatible endpoint, so langchain-openai works. Any
# other OpenAI-compatible server (OpenAI, OpenRouter, Ollama, vLLM) can be used
# by setting LLM_BASE_URL + LLM_API_KEY.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY") or GEMINI_API_KEY
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-3.8-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
# Client-side pacing for rate-limited keys (free tiers): max requests per minute
# (0 = unlimited) and how many times to wait-and-retry a per-minute 429.
LLM_RPM = float(os.getenv("LLM_RPM", "0"))
LLM_RATE_LIMIT_RETRIES = int(os.getenv("LLM_RATE_LIMIT_RETRIES", "6"))

# USD per 1M tokens (input, output), for cost estimates. Unknown models -> 0.
# Add current prices for the models you use (e.g. gemini-3.8-flash); these are not auto-updated.
MODEL_PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

# --- Loop ---
MAX_TRIES = int(os.getenv("MAX_TRIES", "3"))
FEEDBACK_LEVEL = os.getenv("FEEDBACK_LEVEL", "full")
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

# --- Sandbox ---
SANDBOX_TIMEOUT_SECONDS = float(os.getenv("SANDBOX_TIMEOUT_SECONDS", "10"))
SANDBOX_PER_TEST_TIMEOUT_SECONDS = float(os.getenv("SANDBOX_PER_TEST_TIMEOUT_SECONDS", "2"))
SANDBOX_MEMORY_LIMIT_MB = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", "512"))

# --- Paths ---
TASKS_PATH = Path(os.getenv("AGENT_EVAL_TASKS", PROJECT_ROOT / "tasks" / "tasks.json"))
# Named task suites: "easy" is the original function-from-docstring set, "hard"
# mixes bug-fixing, stateful classes, long specs and performance limits.
SUITES = {
    "easy": PROJECT_ROOT / "tasks" / "tasks.json",
    "hard": PROJECT_ROOT / "tasks" / "tasks_hard.json",
    # Procedurally generated TRAINING tasks for self-training and the verifier.
    # Never used for evaluation; `python main.py gen-train-tasks` (re)builds it.
    "train": PROJECT_ROOT / "tasks" / "tasks_train.json",
}
DEFAULT_SUITE = os.getenv("AGENT_EVAL_SUITE", "easy")
DB_PATH = Path(os.getenv("AGENT_EVAL_DB", PROJECT_ROOT / "data" / "agent_eval.db"))
REPORTS_DIR = PROJECT_ROOT / "reports"
