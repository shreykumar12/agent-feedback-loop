"""Safely execute untrusted, LLM-generated code and score it deterministically.

This is the primary source of truth for the whole project: correctness comes
from running real tests, never from asking an LLM.

Isolation (v1, documented as NOT strong isolation):
  - fresh ``tempfile.TemporaryDirectory`` per run, used as cwd and HOME
  - separate process: ``python -I -S`` (isolated mode, no site-packages, so
    candidate code gets the stdlib only and ignores PYTHON* env vars)
  - environment stripped to a minimal allowlist -- API keys never reach it
  - POSIX rlimits: CPU seconds, address space (memory), max file size
  - own process group, killed wholesale on the global timeout
Upgrade path: Docker/gVisor/Firecracker per run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_eval import config
from agent_eval.models import Task, TestCaseResult, TestRunResult

HARNESS_PATH = Path(__file__).with_name("_harness.py")
OUTPUT_LIMIT = 4000

_SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _limit_resources(cpu_seconds: int, memory_mb: int, file_mb: int = 10) -> None:
    """Runs in the child between fork and exec (POSIX only)."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    mem = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):  # e.g. macOS refuses RLIMIT_AS
        pass
    size = file_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))


def _truncate(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...<truncated {len(text) - limit} chars>"


def _parse_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn final line from a killed process
    return events


def run_tests(
    code: str,
    task: Task,
    timeout: float | None = None,
    per_test_timeout: float | None = None,
    memory_mb: int | None = None,
) -> TestRunResult:
    """Run ``task.test_code`` against ``code`` in a subprocess. Never raises
    for anything the candidate code does -- every failure is a result."""
    timeout = timeout or config.SANDBOX_TIMEOUT_SECONDS
    per_test_timeout = per_test_timeout or task.per_test_timeout or config.SANDBOX_PER_TEST_TIMEOUT_SECONDS
    memory_mb = memory_mb or config.SANDBOX_MEMORY_LIMIT_MB

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agent_eval_") as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        shutil.copyfile(HARNESS_PATH, workdir / "harness.py")
        results_path = workdir / "results.jsonl"
        (workdir / "config.json").write_text(json.dumps({
            "workdir": str(workdir),
            "results_path": str(results_path),
            "test_code": task.test_code,
            "entry_point": task.entry_point,
            "per_test_timeout": per_test_timeout,
        }), encoding="utf-8")

        env = dict(_SAFE_ENV, HOME=str(workdir), TMPDIR=str(workdir))
        popen_kwargs: dict = {}
        if os.name == "posix":
            cpu = int(timeout) + 1
            popen_kwargs["preexec_fn"] = lambda: _limit_resources(cpu, memory_mb)
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "harness.py", "config.json"],
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            stdout, stderr = proc.communicate()
        events = _parse_events(results_path)

    result = _build_result(
        events,
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
        timed_out=timed_out,
        returncode=proc.returncode,
        timeout=timeout,
    )
    result.duration_s = round(time.perf_counter() - start, 4)
    return result


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _build_result(
    events: list[dict],
    stdout: str,
    stderr: str,
    timed_out: bool,
    returncode: int | None,
    timeout: float,
) -> TestRunResult:
    captured_stdout = ""
    collected: list[str] = []
    cases: dict[str, TestCaseResult] = {}
    done = False
    for event in events:
        kind = event.pop("event", None)
        if kind == "module_error":
            return TestRunResult(
                passed=False,
                stdout=_truncate(stdout),
                stderr=_truncate(stderr),
                error_type=event["error_type"],
                error=_truncate(event["error"], 2000),
                timed_out=event["error_type"] == "timeout",
            )
        if kind == "collected":
            collected = event["names"]
        elif kind == "case":
            cases[event["name"]] = TestCaseResult(**event)
        elif kind == "done":
            done = True
            captured_stdout = event.get("stdout", "")

    stdout = _truncate(captured_stdout + stdout)
    stderr = _truncate(stderr)

    if not done:
        # The process died mid-run: global timeout, rlimit kill, or crash.
        killed_by_limit = os.name == "posix" and returncode == -signal.SIGXCPU
        if timed_out or killed_by_limit:
            error_type = "timeout"
            reason = f"Sandbox exceeded the {timeout}s time limit (infinite loop or too slow?)"
        elif "MemoryError" in stderr:
            error_type = "memory_error"
            reason = "Process ran out of memory"
        else:
            error_type = "crash"
            reason = f"Test process crashed (exit code {returncode})"
        remaining = [n for n in collected if n not in cases]
        if not collected:
            return TestRunResult(
                passed=False, stdout=stdout, stderr=stderr, timed_out=error_type == "timeout",
                error_type=error_type, error=reason,
            )
        for i, name in enumerate(remaining):
            cases[name] = TestCaseResult(
                name=name,
                passed=False,
                error_type=error_type,
                error=reason if i == 0 else "Not run: sandbox stopped before this test",
            )
        ordered = [cases[n] for n in collected]
        return TestRunResult(
            passed=False, cases=ordered, stdout=stdout, stderr=stderr,
            timed_out=error_type == "timeout",
        )

    ordered = [cases[n] for n in collected if n in cases]
    if not ordered:
        return TestRunResult(
            passed=False, stdout=stdout, stderr=stderr,
            error_type="crash", error="No tests were collected for this task.",
        )
    return TestRunResult(
        passed=all(c.passed for c in ordered),
        cases=ordered,
        stdout=stdout,
        stderr=stderr,
        timed_out=any(c.error_type == "timeout" for c in ordered),
    )
