# WHY THIS FILE EXISTS:
#   Engineering pillar #1 and #2 from the spec: safely execute untrusted,
#   LLM-generated code and judge correctness DETERMINISTICALLY by running real
#   tests. This is the primary score for the whole project -- never ask an LLM
#   whether code is correct.
#
# WHAT IT NEEDS:
#   - run_tests(code, task) -> TestRunResult
#   - Write candidate code + the task's test harness into a fresh temp
#     directory (tempfile.TemporaryDirectory) so runs can't touch each other.
#   - Execute with `subprocess.run([sys.executable, ...], timeout=...,
#     capture_output=True)` -- a separate process, never exec()/eval() in-process.
#   - Resource limits (v1): on macOS/Linux use `resource.setrlimit` via
#     `preexec_fn` for CPU time and memory; strip env vars (don't leak
#     GEMINI_API_KEY to generated code); run with cwd set to the temp dir.
#   - Handle every failure mode as a *result*, not an exception:
#       * syntax error / import error
#       * timeout (infinite loops) -> timed_out=True
#       * crash / non-zero exit
#   - Produce PER-TEST results (name, passed, error, expected vs actual).
#     Easiest approach: have the harness run each test case individually and
#     print a JSON summary line to stdout that this module parses.
#   - Upgrade path (build step 8): Docker-per-run for stronger isolation.

from agent_eval.models import Task, TestRunResult


def run_tests(code: str, task: Task) -> TestRunResult:
    raise NotImplementedError
