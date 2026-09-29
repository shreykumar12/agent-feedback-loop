import pytest

from agent_eval import metrics
from tests.helpers import store_run


@pytest.fixture
def summary(tmp_db):
    store_run("r", {
        "e1": [True],
        "e2": [False, True],
        "h1": ["syntax_error", False, True],
        "h2": ["timeout", False, False],
    })
    return metrics.load_summary("r")


def test_headline_metrics(summary):
    assert summary["num_tasks"] == 4
    assert summary["pass_rate"] == pytest.approx(0.75)
    assert summary["pass_at_1"] == pytest.approx(0.25)
    assert summary["pass_at_k"] == pytest.approx([0.25, 0.5, 0.75])
    assert summary["marginal_gain"] == pytest.approx([0.25, 0.25, 0.25])
    assert summary["self_correction_lift"] == pytest.approx(0.5)
    assert summary["recovery_rate"] == pytest.approx(2 / 3)
    assert summary["mean_tries_to_pass"] == pytest.approx(2.0)
    assert summary["tries_distribution"] == {"1": 1, "2": 1, "3": 1, "failed": 1}
    lo, hi = summary["pass_rate_ci95"]
    assert 0 < lo < 0.75 < hi <= 1


def test_error_breakdown(summary):
    assert summary["first_attempt_errors"] == {"wrong_answer": 1, "syntax_error": 1, "timeout": 1}
    assert summary["error_counts"]["wrong_answer"] == 4
    assert summary["transitions"]["syntax_error -> wrong_answer"] == 1
    assert summary["transitions"]["wrong_answer -> passed"] == 2


def test_retry_dynamics(summary):
    rd = summary["retry_dynamics"]
    # 5 retries: e2 (0.5->1), h1 (0->0.5, 0.5->1), h2 (0->0.5, 0.5->0.5)
    assert rd["retries"] == 5
    assert rd["improved"] == 4 and rd["same"] == 1 and rd["regressed"] == 0
    assert rd["stuck"] == 0


def test_partial_credit_is_monotone_with_carry_forward(summary):
    curve = summary["partial_credit"]
    assert curve == sorted(curve)
    assert curve[-1] == pytest.approx((1 + 1 + 1 + 0.5) / 4)


def test_by_difficulty(summary):
    easy, hard = summary["by_difficulty"]["easy"], summary["by_difficulty"]["hard"]
    assert easy["pass_at_1"] == 0.5 and easy["pass_rate"] == 1.0
    assert hard["pass_at_1"] == 0.0 and hard["pass_rate"] == 0.5


def test_tokens_and_cost(summary):
    assert summary["tokens_in"] == 100 * 9
    assert summary["tokens_per_solved"] == pytest.approx(150 * 9 / 3)
    assert metrics.estimate_cost("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)
    assert metrics.estimate_cost("unknown", 10, 10) == 0.0


def test_wilson_interval_edges():
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = metrics.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo > 0.6


def test_aggregate_summaries(tmp_db):
    store_run("a", {"t1": [True], "t2": [False]}, max_tries=1)
    store_run("b", {"t1": [True], "t2": [True]}, max_tries=1, seed=1)
    agg = metrics.aggregate_summaries([metrics.load_summary("a"), metrics.load_summary("b")])
    assert agg["num_runs"] == 2
    assert agg["pass_rate"]["mean"] == pytest.approx(0.75)
    assert agg["pass_rate"]["std"] > 0
