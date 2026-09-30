"""Task suites, categories and the plumbing that keeps easy and hard runs apart."""

import sqlite3

import pytest

from agent_eval import config, prompts, storage
from agent_eval.agent import extract_code
from agent_eval.models import Task
from agent_eval.sandbox import run_tests
from agent_eval.tasks import TaskFormatError, load_tasks, suite_name, suite_path


def test_easy_suite_prompts_render_exactly_as_before():
    # Changing v1's wording would silently change the experiment for existing runs.
    _, user = prompts.render("v1", "def f(): ...")
    assert user == "Implement the following function:\n\n```python\ndef f(): ...\n```"


def test_category_specific_lead_in():
    _, user = prompts.render("v1", "class Cache: ...", category="stateful")
    assert user.startswith("Implement the following class:")
    _, retry = prompts.render("v2", "def f(): ...", previous_code="x", feedback="fb", category="bugfix")
    assert "def f(): ..." in retry and "fb" in retry
    _, user = prompts.render("v1", "def f(): ...", category="bugfix")
    assert "bugs" in user


def test_extract_code_prefers_block_defining_the_class():
    text = "```python\nprint('demo')\n```\n```python\nclass Ledger:\n    pass\n```"
    assert extract_code(text, entry_point="Ledger").startswith("class Ledger")


def test_task_level_per_test_timeout():
    slow = "import time\ndef f():\n    time.sleep(0.6)\n    return 1\n"
    base = dict(task_id="t", prompt="", entry_point="f", test_code="def test_a():\n    assert f() == 1\n")
    assert run_tests(slow, Task(**base)).passed  # default 2 s limit
    tight = run_tests(slow, Task(**base, per_test_timeout=0.3))
    assert tight.cases[0].error_type == "timeout"


def test_suite_resolution(tmp_path):
    assert suite_path("easy") == config.SUITES["easy"]
    assert suite_name(None) == config.DEFAULT_SUITE
    custom = tmp_path / "mine.json"
    custom.write_text("[]")
    assert suite_path(str(custom)) == custom and suite_name(str(custom)) == "mine"
    with pytest.raises(TaskFormatError):
        suite_path("no-such-suite")


def test_unknown_category_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"task_id": "x", "prompt": "p", "entry_point": "f", "test_code": "t", "category": "vibes"}]')
    with pytest.raises(TaskFormatError):
        load_tasks(path)


def test_old_database_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
            feedback_level TEXT NOT NULL DEFAULT 'full', max_tries INTEGER NOT NULL, timestamp TEXT NOT NULL,
            suite_hash TEXT NOT NULL DEFAULT '', seed INTEGER NOT NULL DEFAULT 0, notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running', finished_at TEXT);
        CREATE TABLE tasks (task_id TEXT PRIMARY KEY, prompt TEXT NOT NULL, entry_point TEXT NOT NULL,
            test_code TEXT NOT NULL, difficulty TEXT NOT NULL DEFAULT 'medium', tags TEXT NOT NULL DEFAULT '[]');
        INSERT INTO runs (run_id, model, prompt_version, max_tries, timestamp) VALUES ('old', 'm', 'v1', 3, 't');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    storage.init_db()
    storage.init_db()  # idempotent
    [run] = storage.list_runs()
    assert run["suite"] == "easy"  # pre-suite history counts as the easy suite
    storage.save_task(Task(task_id="t", prompt="p", entry_point="f", test_code="x", category="bugfix"))
    assert storage.get_task("t")["category"] == "bugfix"


def test_runs_record_their_suite_and_selectors_filter_by_it(tmp_db):
    from agent_eval.run_suite import run_suite

    easy_id = run_suite(model="sim-strong", prompt_version="v1", task_ids=["caesar_shift"], quiet=True)
    assert storage.get_run(easy_id)["suite"] == "easy"
    assert storage.resolve_run_refs("@sim-strong/full/v1/easy") == [easy_id]
    with pytest.raises(KeyError):
        storage.resolve_run_refs("@sim-strong/full/v1/hard")
