"""The offline simulated agent must behave like a plausible model: deterministic
per seed, better when stronger, and never worse when given more information."""

import statistics

import pytest

from agent_eval.graph import LoopSettings, run_task
from agent_eval.simulated import choose_program, extract_probes
from agent_eval.tasks import load_tasks

TASKS = load_tasks()
SUBSET = [t for t in TASKS if t.task_id in {
    "caesar_shift", "rounded_mean", "merge_intervals", "rank_players", "window_maxima",
    "summarize_ranges", "lru_simulate", "evaluate_expression", "parse_csv_line", "course_order"}]


def _pass_rate(model, level, seeds=3):
    rates = []
    for seed in range(seeds):
        results = [run_task(t, LoopSettings(model=model, feedback_level=level, max_tries=3, seed=seed))[0]
                   for t in SUBSET]
        rates.append(sum(r.passed for r in results) / len(results))
    return statistics.fmean(rates)


def test_deterministic_per_seed():
    task = TASKS[0]
    a = choose_program("sim-base", task, None, [], 1, seed=3)
    b = choose_program("sim-base", task, None, [], 1, seed=3)
    assert a == b


@pytest.mark.slow
def test_more_feedback_information_never_hurts():
    minimal = _pass_rate("sim-weak", "minimal")
    names = _pass_rate("sim-weak", "names")
    full = _pass_rate("sim-weak", "full")
    assert minimal <= names <= full
    assert full > minimal


def test_stronger_profile_has_higher_first_try_rate():
    def first_try(model):
        return statistics.fmean(
            choose_program(model, t, None, [], 1, seed=s) == t.canonical_solution
            for t in TASKS for s in range(10))
    assert first_try("sim-strong") > first_try("sim-base") > first_try("sim-weak")


def test_extract_probes_only_takes_rerunnable_assertions():
    feedback = ("1/3 tests passed. Failing tests:\n- test_a\n    test line: assert f(1) == 2\n"
                "- test_b\n    test line: assert False, \"expected ValueError\"\n")
    assert extract_probes(feedback) == ["assert f(1) == 2"]
