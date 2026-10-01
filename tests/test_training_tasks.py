"""Tests for the procedural training-task generator (agent_eval.training_tasks)."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

import pytest

from agent_eval import training_tasks
from agent_eval.tasks import load_tasks
from agent_eval.training_tasks import (
    FAMILIES,
    generate_training_tasks,
    validate_generated,
    write_training_suite,
)


@pytest.fixture(scope="module")
def hundred() -> list[dict]:
    return generate_training_tasks(100)


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _code_shape(task: dict) -> str:
    """Canonical solution without docstrings, function renamed to F."""
    tree = ast.parse(task["canonical_solution"])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == task["entry_point"]:
                node.name = "F"
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_generation_is_deterministic():
    a = generate_training_tasks(80, seed=3)
    b = generate_training_tasks(80, seed=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    sub = generate_training_tasks(10, seed=3, families=["slugify", "bfs_hops"])
    assert sub == generate_training_tasks(10, seed=3, families=["slugify", "bfs_hops"])


def test_different_seeds_give_different_tasks():
    a = generate_training_tasks(76, seed=0)
    b = generate_training_tasks(76, seed=1)
    differing = sum(x["test_code"] != y["test_code"] for x, y in zip(a, b, strict=True))
    assert differing > 0.9 * len(a)


def test_at_least_25_families_all_represented(hundred):
    assert len(FAMILIES) >= 25
    seen = {t["tags"][1] for t in hundred}
    assert seen == set(FAMILIES)


def test_task_schema(hundred):
    for t in hundred:
        family = t["tags"][1]
        assert t["task_id"].startswith(f"gen_{family}_")
        assert t["category"] == "function"
        assert t["difficulty"] in ("easy", "medium", "hard")
        assert t["tags"][0] == "generated"
        assert t["prompt"].startswith(f"def {t['entry_point']}(")
        assert t["canonical_solution"].startswith(t["prompt"])
        assert 2 <= len(t["mutants"]) <= 4
        assert len(set(t["mutants"])) == len(t["mutants"])
        assert t["canonical_solution"] not in t["mutants"]
        tests = re.findall(r"^def (test_\w+)\(\):", t["test_code"], flags=re.M)
        assert 5 <= len(tests) <= 8
        assert len(set(tests)) == len(tests)
        assert ">>>" in t["prompt"]
        ast.parse(t["test_code"])
        for m in t["mutants"]:
            ast.parse(m)


def test_unique_ids_and_no_overlap_with_eval_suites(hundred):
    many = generate_training_tasks(400)
    ids = [t["task_id"] for t in many]
    assert len(ids) == len(set(ids))
    eval_tasks = load_tasks(suite="easy") + load_tasks(suite="hard")
    eval_ids = {t.task_id for t in eval_tasks}
    eval_prompts = {_normalize_prompt(t.prompt) for t in eval_tasks}
    eval_entry_points = {t.entry_point for t in eval_tasks}
    for t in many:
        assert t["task_id"] not in eval_ids
        assert _normalize_prompt(t["prompt"]) not in eval_prompts
        assert t["entry_point"] not in eval_entry_points


def test_generated_tasks_load_through_load_tasks(tmp_path, hundred):
    path = tmp_path / "train.json"
    path.write_text(json.dumps(hundred, indent=2), encoding="utf-8")
    loaded = load_tasks(path)
    assert [t.task_id for t in loaded] == [t["task_id"] for t in hundred]
    assert all(t.canonical_solution and t.mutants for t in loaded)


def test_sample_validates_in_sandbox():
    # Two tasks per family: canonical passes every test, every mutant is killed.
    sample = generate_training_tasks(2 * len(FAMILIES), seed=7)
    assert validate_generated(sample, workers=4) == []


def test_validate_reports_broken_tasks():
    good = generate_training_tasks(2, seed=0)
    broken = dict(good[0], task_id="broken_canonical",
                  canonical_solution=good[0]["mutants"][0])
    survivor = dict(good[1], task_id="surviving_mutant",
                    mutants=[good[1]["canonical_solution"], *good[1]["mutants"]])
    problems = validate_generated([broken, survivor], workers=2)
    assert any(p.startswith("broken_canonical: canonical failed") for p in problems)
    assert any(p.startswith("surviving_mutant: mutant 0 survived") for p in problems)


def test_variation_changes_the_code_not_just_names():
    no_variation = []
    for family in FAMILIES:
        tasks = generate_training_tasks(6, seed=0, families=[family])
        names = {t["entry_point"] for t in tasks}
        shapes = {_code_shape(t) for t in tasks}
        if len(shapes) < 2:
            no_variation.append(family)
        assert len({t["test_code"] for t in tasks}) == 6, family
        assert names  # sanity
    assert no_variation == []


def test_write_training_suite_drops_invalid_tasks(tmp_path, monkeypatch):
    real = training_tasks.generate_training_tasks

    def with_a_broken_one(n, seed=0, families=None):
        tasks = real(n, seed=seed, families=families)
        tasks[0] = dict(tasks[0], canonical_solution=tasks[0]["mutants"][0])
        return tasks

    monkeypatch.setattr(training_tasks, "generate_training_tasks", with_a_broken_one)
    path = tmp_path / "out" / "train.json"
    stats = write_training_suite(path, 6, seed=2)
    assert stats["written"] == 5 and stats["dropped"] == 1
    written = json.loads(path.read_text(encoding="utf-8"))
    assert len(written) == 5
    counts = defaultdict(int)
    for t in written:
        counts[t["tags"][1]] += 1
    assert stats["families"] == dict(counts)
    assert len(load_tasks(path)) == 5


def test_write_without_validation(tmp_path):
    stats = write_training_suite(tmp_path / "t.json", 40, seed=5, validate=False)
    assert stats["written"] == 40 and stats["dropped"] == 0
    assert sum(stats["families"].values()) == 40


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        generate_training_tasks(3, families=["no_such_family"])
