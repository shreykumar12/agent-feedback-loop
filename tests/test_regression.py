"""The comparator decides what "got worse" means. A bug here silently hides
regressions, which defeats the point of the system."""

import pytest

from agent_eval import regression
from tests.helpers import store_run


@pytest.fixture
def two_runs(tmp_db):
    store_run("base", {
        "t1": [True],                 # stays passing
        "t2": [False, True],          # passes in both, needs more tries in candidate
        "t3": [True],                 # newly failing in candidate
        "t4": [False, False, False],  # newly passing in candidate
        "t5": [True],                 # only in baseline
    })
    store_run("cand", {
        "t1": [True],
        "t2": [False, False, True],
        "t3": [False, False, False],
        "t4": [False, True],
        "t6": [True],                 # only in candidate
    }, model="other", seed=1)
    return regression.compare_runs("base", "cand")


def test_detects_newly_failing_task(two_runs):
    assert two_runs["newly_failing"] == ["t3"]
    assert two_runs["newly_passing"] == ["t4"]
    assert two_runs["regression"]
    assert any("newly failing" in r for r in two_runs["reasons"])


def test_more_tries(two_runs):
    assert two_runs["more_tries"] == [{"task_id": "t2", "baseline": 2, "candidate": 3}]
    assert two_runs["fewer_tries"] == []


def test_pass_rate_delta(two_runs):
    agg = two_runs["aggregate"]
    # common tasks t1..t4: baseline passes 3/4, candidate passes 3/4
    assert agg["pass_rate"]["baseline"] == pytest.approx(0.75)
    assert agg["pass_rate"]["candidate"] == pytest.approx(0.75)
    assert agg["pass_rate"]["delta"] == pytest.approx(0.0)
    # pass@1: baseline t1,t3 = 2/4; candidate t1 = 1/4
    assert agg["pass_at_1"]["delta"] == pytest.approx(-0.25)
    # mean tries among passed: baseline (1+2+1)/3, candidate (1+3+2)/3
    assert agg["mean_tries_to_pass"]["delta"] == pytest.approx(2 / 3)


def test_tasks_in_only_one_run_are_reported(two_runs):
    assert two_runs["only_in_baseline"] == ["t5"]
    assert two_runs["only_in_candidate"] == ["t6"]
    assert two_runs["num_common"] == 4
    assert any("task sets differ" in w for w in two_runs["warnings"])


def test_changed_config_is_reported(two_runs):
    assert two_runs["changed"]["model"] == ("m", "other")


def test_identical_runs_are_not_a_regression(tmp_db):
    outcomes = {"t1": [True], "t2": [False, True]}
    store_run("a", outcomes)
    store_run("b", outcomes, seed=1)
    comparison = regression.compare_runs("a", "b")
    assert not comparison["regression"]
    assert comparison["mcnemar_p"] == 1.0
    assert "no regression" in regression.format_report(comparison)


def test_suite_hash_mismatch_warns(tmp_db):
    store_run("a", {"t1": [True]}, suite_hash="one")
    store_run("b", {"t1": [True]}, suite_hash="two", seed=1)
    assert any("suite hash" in w for w in regression.compare_runs("a", "b")["warnings"])


def test_mcnemar_exact():
    assert regression.mcnemar_exact(0, 0) == 1.0
    assert regression.mcnemar_exact(3, 3) == 1.0
    # 8 discordant pairs all one way: p = 2 * 0.5**8
    assert regression.mcnemar_exact(8, 0) == pytest.approx(2 / 256)


def test_bootstrap_ci_contains_observed_delta():
    base = [1] * 10 + [0] * 10
    cand = [1] * 5 + [0] * 15
    lo, hi = regression.paired_bootstrap_ci(base, cand, samples=2000)
    assert lo <= -0.25 <= hi and hi <= 0


def test_report_formats(two_runs):
    text = regression.format_report(two_runs)
    assert "REGRESSION" in text and "t3" in text and "McNemar" in text
