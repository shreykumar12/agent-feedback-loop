"""Offline, deterministic simulated coding agent (``--model sim-*``).

Purpose: exercise the *entire* pipeline -- prompts, extraction, sandbox, loop,
storage, metrics, regression detection, dashboard -- with no API key and no
cost, reproducibly. Results from sim models are a demonstration of the
measurement system, NOT claims about any real LLM.

How it behaves. Each task ships a canonical solution and a list of plausible
buggy mutants (ordered most -> least plausible). The simulated model "writes"
one of those programs:

  * First attempt: correct with probability ``first_try[difficulty]`` of the
    model profile, otherwise a mutant (earlier mutants are likelier).
  * Retry: a failure is evidence the task is hard *for this model*, so a
    blind retry succeeds with only ``p_blind = first_try * BLIND_FACTOR``.
    Feedback can only add to that floor; how much depends on what the
    feedback *text* contains:
      - failing assertion lines ("full" feedback): it turns each shown
        assertion into a probe test and runs every untried candidate against
        the probes in the sandbox, discarding candidates that fail them --
        a genuine use of the feedback's information. It then submits the
        correct program with probability ``p_blind + (1 - p_blind) * repair``
        (else another surviving candidate).
      - module-level diagnosis (syntax/import error explained): same
        probability, without probes.
      - failing test names only: ``p_blind + (1 - p_blind) * repair * NAMES_FACTOR``.
      - no usable information ("minimal"/"none"): ``p_blind``, and it may
        resubmit its previous answer unchanged (a "stuck" retry).

Token counts are estimated from the real rendered prompts (~4 chars/token);
latency is simulated (not slept) so a suite run takes seconds.
"""

from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass

from agent_eval.models import Task

NAMES_FACTOR = 0.45
BLIND_FACTOR = 0.5
STUCK_PROB = 0.35
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class SimProfile:
    first_try: dict[str, float]
    repair: float
    latency_range: tuple[float, float]


PROFILES = {
    "sim-strong": SimProfile({"easy": 0.93, "medium": 0.75, "hard": 0.5}, repair=0.85, latency_range=(1.5, 5.0)),
    "sim-base": SimProfile({"easy": 0.82, "medium": 0.55, "hard": 0.3}, repair=0.7, latency_range=(0.6, 2.5)),
    "sim-weak": SimProfile({"easy": 0.65, "medium": 0.35, "hard": 0.15}, repair=0.45, latency_range=(0.3, 1.2)),
}


def get_profile(model: str) -> SimProfile:
    return PROFILES.get(model, PROFILES["sim-base"])


def _rng(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def _pick_mutant(rng: random.Random, mutants: list[str], exclude: set[str]) -> str | None:
    pool = [m for m in mutants if m not in exclude]
    if not pool:
        return None
    weights = [1.0 / (i + 1) for i in range(len(pool))]  # earlier = more plausible
    return rng.choices(pool, weights=weights, k=1)[0]


_TEST_LINE = re.compile(r"^\s*test line: (.*)$", re.MULTILINE)


def extract_probes(feedback: str) -> list[str]:
    """Assertion statements shown in 'full' feedback that can be re-run as probes."""
    probes = []
    for line in _TEST_LINE.findall(feedback or ""):
        line = line.strip()
        if line.startswith("assert") and not line.startswith("assert False"):
            probes.append(line)
    return probes


def _probe_survivors(task: Task, candidates: list[str], probes: list[str]) -> list[str]:
    from agent_eval.sandbox import run_tests

    probe_code = "\n\n".join(
        f"def test_probe_{i}():\n{textwrap.indent(p, '    ')}" for i, p in enumerate(probes)
    )
    probe_task = Task(task_id=f"{task.task_id}__probe", prompt="", entry_point=task.entry_point,
                      test_code=probe_code)
    matrix = {c: run_tests(c, probe_task, timeout=5, per_test_timeout=1) for c in candidates}
    # Drop probes nobody passes (e.g. they reference locals from the full test).
    informative = {
        i for i in range(len(probes))
        if any(r.cases and i < len(r.cases) and r.cases[i].passed for r in matrix.values())
    }
    survivors = []
    for cand, result in matrix.items():
        if not result.cases:
            continue  # candidate does not even import -> rejected
        if all(result.cases[i].passed for i in informative if i < len(result.cases)):
            survivors.append(cand)
    return survivors


def choose_program(model: str, task: Task, feedback: str | None, history: list[str],
                   attempt_number: int, seed: int) -> str:
    profile = get_profile(model)
    rng = _rng(model, seed, task.task_id, attempt_number)
    correct = task.canonical_solution or ""
    mutants = list(task.mutants)
    p_first = profile.first_try.get(task.difficulty, profile.first_try["medium"])

    if attempt_number == 1 or feedback is None:
        if not mutants or rng.random() < p_first:
            return correct
        return _pick_mutant(rng, mutants, set()) or correct

    tried = set(history)
    p_blind = p_first * BLIND_FACTOR
    p_informed = p_blind + (1 - p_blind) * profile.repair
    probes = extract_probes(feedback)
    if probes:
        untried = [c for c in [correct, *mutants] if c not in tried]
        survivors = _probe_survivors(task, untried, probes) if untried else []
        if correct in survivors and (rng.random() < p_informed or len(survivors) == 1):
            return correct
        others = [s for s in survivors if s != correct]
        if others:
            return rng.choice(others)
        return correct if rng.random() < p_informed else (_pick_mutant(rng, mutants, set()) or correct)

    if "Only the Python standard library" in feedback or "not valid Python" in feedback \
            or "function is not defined" in feedback or "at import time" in feedback:
        if rng.random() < p_informed:
            return correct
        return _pick_mutant(rng, mutants, tried) or correct

    if "Failing tests:" in feedback:  # names-level feedback
        if rng.random() < p_blind + (1 - p_blind) * profile.repair * NAMES_FACTOR:
            return correct
        return _pick_mutant(rng, mutants, tried) or history[-1]

    # No usable information: a blind retry, often anchored to the previous answer.
    if rng.random() < p_blind:
        return correct
    if history and rng.random() < STUCK_PROB:
        return history[-1]
    return _pick_mutant(rng, mutants, set()) or correct

def respond(model: str, task: Task, system: str, user: str, *, feedback: str | None,
            history: list[str], attempt_number: int, seed: int) -> tuple[str, int, int, float]:
    """Return (raw_response, tokens_in, tokens_out, simulated_latency_s)."""
    program = choose_program(model, task, feedback, history, attempt_number, seed)
    raw = f"```python\n{program.strip()}\n```"
    tokens_in = (len(system) + len(user)) // CHARS_PER_TOKEN
    tokens_out = len(raw) // CHARS_PER_TOKEN
    lo, hi = get_profile(model).latency_range
    latency = _rng("latency", model, seed, task.task_id, attempt_number).uniform(lo, hi)
    return raw, tokens_in, tokens_out, latency
