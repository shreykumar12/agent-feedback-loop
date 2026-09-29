"""Smoke tests for the Streamlit dashboard: the script must render every
section without raising, on an empty DB and on a seeded multi-config DB."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent_eval import config, storage
from tests.helpers import store_run

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def _point_at(monkeypatch, path: Path) -> None:
    # config reads AGENT_EVAL_DB at import time; storage reads config.DB_PATH at call time.
    monkeypatch.setenv("AGENT_EVAL_DB", str(path))
    monkeypatch.setattr(config, "DB_PATH", path)


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60).run()


def _assert_ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_empty_db_shows_hint(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path / "empty.db")
    at = _run_app()
    _assert_ok(at)
    assert any("python main.py run --model sim-base" in i.value for i in at.info)


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path / "seeded.db")
    storage.init_db()
    # Two configurations x two seeds; the second config is worse on hard tasks.
    for seed in (0, 1):
        store_run(f"base{seed}", {"e1": [True], "e2": [False, True], "h1": [False, "runtime_error", True],
                                  "h2": [True]},
                  model="sim-base", seed=seed, feedback_level="full")
        store_run(f"cand{seed}", {"e1": [True], "e2": [False, False, False], "h1": ["syntax_error", False, False],
                                  "h2": [False, True]},
                  model="sim-weak", seed=seed, feedback_level="none")
    return tmp_path


def test_seeded_db_renders_all_sections(seeded_db):
    at = _run_app()
    _assert_ok(at)
    assert any("simulated agent" in w.value.lower() for w in at.warning)
    labels = [m.label for m in at.metric]
    for expected in ("Final pass rate", "pass@1", "Self-correction lift", "Recovery rate",
                     "Mean tries-to-pass", "Tokens / solved task"):
        assert expected in labels
    # Default comparison is config vs config (compare_groups) -> a verdict is shown.
    assert at.success or at.error


def test_seeded_db_interactions(seeded_db):
    at = _run_app()
    _assert_ok(at)

    at.selectbox(key="run_select").set_value("cand0").run()
    _assert_ok(at)
    task_box = next(s for s in at.selectbox if s.key.startswith("task_select_cand0"))
    task_box.set_value("h1").run()
    _assert_ok(at)
    assert any("syntax_error" in e.value for e in at.error)

    at.toggle(key="config_view").set_value(True).run()
    _assert_ok(at)
    assert any("±" in str(m.value) for m in at.metric)

    at.radio(key="cmp_mode").set_value("Single runs").run()
    _assert_ok(at)
    at.selectbox(key="cmp_base_run").set_value("base0").run()
    at.selectbox(key="cmp_cand_run").set_value("cand0").run()
    _assert_ok(at)
    assert any("REGRESSION" in e.value for e in at.error)

    at.radio(key="matrix_metric").set_value("first-try solve rate").run()
    at.radio(key="ablation_metric").set_value("pass@1").run()
    _assert_ok(at)


@pytest.mark.slow
@pytest.mark.skipif(not (Path(config.PROJECT_ROOT) / "data" / "agent_eval.db").exists(),
                    reason="no local results DB")
def test_real_db_renders(monkeypatch):
    _point_at(monkeypatch, Path(config.PROJECT_ROOT) / "data" / "agent_eval.db")
    _assert_ok(_run_app())
