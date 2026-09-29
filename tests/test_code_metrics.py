from agent_eval.code_metrics import analyze, edit_ratio


def test_analyze_counts_branches():
    code = "def f(x):\n    if x > 0 and x < 10:\n        for i in range(x):\n            pass\n    return x\n"
    m = analyze(code)
    assert m["cyclomatic_complexity"] == 4  # 1 + if + and + for
    assert m["num_functions"] == 1
    assert m["max_nesting"] == 3
    assert m["sloc"] == 5


def test_analyze_syntax_error_returns_none():
    assert analyze("def f(:\n") is None


def test_edit_ratio():
    assert edit_ratio("a\nb\n", "a\nb\n") == 0.0
    assert edit_ratio("a\n", "z\n") == 1.0
    assert 0 < edit_ratio("a\nb\nc\n", "a\nb\nd\n") < 1
