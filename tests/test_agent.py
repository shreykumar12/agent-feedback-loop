from agent_eval.agent import extract_code, generate
from agent_eval.models import Task
from agent_eval.prompts import PROMPTS, render


def test_extracts_python_fence():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope it helps"
    assert extract_code(text) == "def f():\n    return 1\n"


def test_prefers_block_defining_entry_point():
    text = "```python\nprint('usage')\n```\n\n```python\ndef solve(x):\n    return x\n```"
    assert "def solve" in extract_code(text, entry_point="solve")


def test_falls_back_to_generic_fence_and_raw_text():
    assert extract_code("```\ndef f(): pass\n```") == "def f(): pass\n"
    assert extract_code("def f(): pass") == "def f(): pass\n"


def test_unclosed_fence_does_not_crash():
    assert "def f" in extract_code("```python\ndef f():\n    return 1")


def test_empty_response():
    assert extract_code("") == ""
    assert extract_code(None) == ""


def test_prompt_versions_render():
    for version in PROMPTS:
        system, user = render(version, "def f(): ...")
        assert "def f(): ..." in user and system
        _, retry = render(version, "def f(): ...", previous_code="def f(): return 0", feedback="1/2 tests passed")
        assert "def f(): return 0" in retry and "1/2 tests passed" in retry


def test_simulated_model_needs_no_key():
    task = Task(task_id="t", prompt="def f():\n    ...\n", entry_point="f", test_code="",
                canonical_solution="def f():\n    return 1\n", mutants=[], difficulty="easy")
    gen = generate(task, model="sim-strong", seed=0)
    assert gen.code == "def f():\n    return 1\n"
    assert gen.tokens_in > 0 and gen.tokens_out > 0
