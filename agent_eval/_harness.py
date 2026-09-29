"""Test harness executed INSIDE the sandbox subprocess. Stdlib only.

It is copied into a fresh temp directory next to ``solution.py`` and run as
``python -I -S harness.py <config.json>``. It never runs in the parent process.

Protocol: results are appended as JSON lines to the results file named in the
config, one event per line, so partial results survive if the parent has to
kill the process on the global timeout:

    {"event": "module_error", "error_type": ..., "error": ...}
    {"event": "collected", "names": [...]}
    {"event": "case", "name": ..., "passed": ..., ...}
    {"event": "done"}

Every ``assert <actual> == <expected>`` in the test code is rewritten (AST) into
a helper call so failures report expected vs actual values, like pytest does.
"""

import ast
import importlib.util
import io
import json
import signal
import sys
import time
import traceback

REPR_LIMIT = 500
TEST_FILENAME = "<tests>"
SOLUTION_MODULE = "solution"


class _Timeout(BaseException):
    """BaseException so candidate code's `except Exception` can't swallow it."""


class _EqFailure(AssertionError):
    def __init__(self, actual, expected):
        super().__init__("values differ")
        self.actual = actual
        self.expected = expected


def _short_repr(value):
    try:
        text = repr(value)
    except Exception as exc:  # a broken __repr__ in candidate code
        text = f"<unrepresentable: {type(exc).__name__}>"
    if len(text) > REPR_LIMIT:
        text = text[:REPR_LIMIT] + "...<truncated>"
    return text


def _ae_eq(actual, expected):
    if not (actual == expected):
        raise _EqFailure(actual, expected)


class _RewriteEqAsserts(ast.NodeTransformer):
    """`assert a == b[, msg]` -> `__ae_eq__(a, b)`; other asserts untouched."""

    def visit_Assert(self, node):
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__ae_eq__", ctx=ast.Load()),
                    args=[test.left, test.comparators[0]],
                    keywords=[],
                )
            )
            return ast.copy_location(call, node)
        return node


def _statement_source(test_source, tree, lineno):
    """Source of the innermost statement in the test code covering `lineno`."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                if best is None or node.lineno >= best.lineno:
                    best = node
    if best is not None:
        segment = ast.get_source_segment(test_source, best)
        if segment:
            return segment.strip()
    lines = test_source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return None


def _test_lineno(tb):
    lineno = None
    for frame, line in traceback.walk_tb(tb):
        if frame.f_code.co_filename == TEST_FILENAME:
            lineno = line
    return lineno


def _format_candidate_traceback(exc):
    """Traceback limited to frames from the candidate solution and the test."""
    frames = [
        f
        for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.endswith("solution.py") or f.filename == TEST_FILENAME
    ]
    for f in frames:
        if f.filename.endswith("solution.py"):
            f.filename = "solution.py"
    lines = ["Traceback (most recent call last):\n"]
    lines += traceback.format_list(frames)
    lines += traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip()


def _classify_failure(case, exc, per_test_timeout):
    if isinstance(exc, _Timeout):
        case["error_type"] = "timeout"
        case["error"] = f"Test exceeded {per_test_timeout}s (infinite loop or too slow?)"
    elif isinstance(exc, _EqFailure):
        case["error_type"] = "wrong_answer"
        case["expected"] = _short_repr(exc.expected)
        case["actual"] = _short_repr(exc.actual)
        case["error"] = f"expected {case['expected']}, got {case['actual']}"
    elif isinstance(exc, AssertionError):
        case["error_type"] = "wrong_answer"
        case["error"] = f"AssertionError: {exc}" if str(exc) else "AssertionError"
    elif isinstance(exc, MemoryError):
        case["error_type"] = "memory_error"
        case["error"] = "MemoryError (memory limit exceeded)"
    elif isinstance(exc, RecursionError):
        case["error_type"] = "runtime_error"
        case["error"] = f"RecursionError: {exc}"
    else:
        case["error_type"] = "runtime_error"
        case["error"] = _format_candidate_traceback(exc)


def _on_alarm(signum, frame):
    raise _Timeout()


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        config = json.load(fh)
    workdir = config["workdir"]
    results = open(config["results_path"], "a", encoding="utf-8")

    def emit(event):
        results.write(json.dumps(event) + "\n")
        results.flush()

    sys.path.insert(0, workdir)
    captured = io.StringIO()
    real_stdout = sys.stdout
    has_alarm = hasattr(signal, "setitimer")
    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)

    # 1. Syntax check (before import, for a precise message).
    solution_path = f"{workdir}/solution.py"
    with open(solution_path, encoding="utf-8") as fh:
        source = fh.read()
    try:
        compile(source, "solution.py", "exec")
    except SyntaxError as exc:
        emit({
            "event": "module_error",
            "error_type": "syntax_error",
            "error": f"SyntaxError: {exc.msg} (line {exc.lineno})\n{(exc.text or '').rstrip()}",
        })
        return

    # 2. Import the candidate module (module-level code runs here).
    sys.stdout = captured
    try:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, config["per_test_timeout"])
        spec = importlib.util.spec_from_file_location(SOLUTION_MODULE, solution_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[SOLUTION_MODULE] = module
        spec.loader.exec_module(module)
    except _Timeout:
        emit({"event": "module_error", "error_type": "timeout",
              "error": "Timed out while importing the module (module-level code runs forever?)"})
        return
    except ImportError as exc:
        emit({"event": "module_error", "error_type": "import_error",
              "error": _format_candidate_traceback(exc)})
        return
    except MemoryError:
        emit({"event": "module_error", "error_type": "memory_error",
              "error": "MemoryError while importing the module"})
        return
    except BaseException as exc:
        emit({"event": "module_error", "error_type": "runtime_error",
              "error": _format_candidate_traceback(exc)})
        return
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        sys.stdout = real_stdout

    entry = getattr(module, config["entry_point"], None)
    if not callable(entry):
        emit({"event": "module_error", "error_type": "missing_entry_point",
              "error": f"No callable named '{config['entry_point']}' is defined."})
        return

    # 3. Load the (rewritten) tests into a namespace seeded with the solution.
    test_source = config["test_code"]
    tree = ast.parse(test_source, filename=TEST_FILENAME)
    rewritten = ast.fix_missing_locations(_RewriteEqAsserts().visit(ast.parse(test_source)))
    namespace = dict(vars(module))
    namespace["__ae_eq__"] = _ae_eq
    namespace["__name__"] = "__tests__"
    exec(compile(rewritten, TEST_FILENAME, "exec"), namespace)
    tests = [
        (name, fn) for name, fn in namespace.items()
        if name.startswith("test_") and callable(fn)
    ]
    emit({"event": "collected", "names": [name for name, _ in tests]})

    # 4. Run each test in isolation with its own timeout.
    for name, fn in tests:
        case = {"event": "case", "name": name, "passed": False, "error": None,
                "error_type": None, "expected": None, "actual": None, "assertion": None}
        start = time.perf_counter()
        sys.stdout = captured
        try:
            if has_alarm:
                signal.setitimer(signal.ITIMER_REAL, config["per_test_timeout"])
            fn()
            case["passed"] = True
        except BaseException as exc:  # includes SystemExit from candidate code
            if has_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
            _classify_failure(case, exc, config["per_test_timeout"])
            lineno = _test_lineno(exc.__traceback__)
            if lineno is not None:
                case["assertion"] = _statement_source(test_source, tree, lineno)
        finally:
            if has_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
            sys.stdout = real_stdout
        case["duration_s"] = round(time.perf_counter() - start, 6)
        emit(case)

    out = captured.getvalue()
    emit({"event": "done", "stdout": out[-4000:]})


if __name__ == "__main__":
    main()
