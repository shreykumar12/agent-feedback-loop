"""Run the whole task suite under one fixed configuration = one run_id.

A run is the unit of comparison for regression detection. Tasks execute in a
thread pool (LLM calls are I/O bound; each sandbox is its own process); all
DB writes happen on the calling thread. An unexpected per-task exception is
recorded as a failed task with its error and the suite keeps going.
"""

from __future__ import annotations

import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from agent_eval import agent, config, judge, metrics, storage
from agent_eval.graph import LoopSettings, run_task
from agent_eval.models import RunInfo, Task, TaskResult
from agent_eval.tasks import load_tasks, suite_hash


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _execute(task: Task, settings: LoopSettings, use_judge: bool):
    try:
        result, attempts = run_task(task, settings)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return TaskResult(task_id=task.task_id, passed=False, tries_taken=0, final_code="",
                          error=detail), [], traceback.format_exc()
    if use_judge and result.passed:
        result.quality_score = judge.score_quality(task, result.final_code)
    return result, attempts, None


def run_suite(
    model: str,
    prompt_version: str,
    task_ids: list[str] | None = None,
    use_judge: bool = False,
    feedback_level: str | None = None,
    max_tries: int | None = None,
    seed: int = 0,
    workers: int = 4,
    notes: str = "",
    quiet: bool = False,
) -> str:
    if not agent.is_simulated(model) and not config.LLM_API_KEY:
        raise agent.AgentError(
            f"No API key for {model!r}. Set GEMINI_API_KEY (or LLM_API_KEY) in .env, "
            "or run offline with a simulated model: --model sim-base"
        )
    storage.init_db()
    tasks = load_tasks(task_ids=task_ids)
    settings = LoopSettings(
        model=model,
        prompt_version=prompt_version,
        feedback_level=feedback_level or config.FEEDBACK_LEVEL,
        max_tries=max_tries or config.MAX_TRIES,
        seed=seed,
    )
    run = RunInfo(
        run_id=uuid.uuid4().hex,
        model=model,
        prompt_version=prompt_version,
        timestamp=_now(),
        feedback_level=settings.feedback_level,
        max_tries=settings.max_tries,
        suite_hash=suite_hash(tasks),
        seed=seed,
        notes=notes,
    )
    storage.create_run(run)
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))
    log(f"run {run.run_id[:8]}: {model} / prompt {prompt_version} / feedback={settings.feedback_level}"
        f" / max_tries={settings.max_tries} / seed={seed} / {len(tasks)} tasks")

    status = "completed"
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_execute, t, settings, use_judge): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                result, attempts, tb = future.result()
                storage.save_task_run(run.run_id, task, result, attempts)
                mark = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
                trail = " ".join(
                    "ok" if a.test_result.passed else (a.test_result.primary_error_type or "fail")
                    for a in attempts
                )
                log(f"  [{i:>2}/{len(tasks)}] {mark:<5} {task.task_id:<28} tries={result.tries_taken}  {trail}"
                    + (f"  {result.error}" if result.error else ""))
                if tb and not quiet:
                    log(tb)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        storage.finish_run(run.run_id, _now(), status)

    if not quiet:
        s = metrics.load_summary(run.run_id)
        lo, hi = s["pass_rate_ci95"]
        mean_tries = f"{s['mean_tries_to_pass']:.2f}" if s["mean_tries_to_pass"] else "-"
        failed = [r["task_id"] for r in storage.get_results(run.run_id) if not r["passed"]]
        log(f"pass rate {s['pass_rate']:.1%} (95% CI {lo:.0%}-{hi:.0%}) | pass@1 {s['pass_at_1']:.1%} | "
            f"lift {s['self_correction_lift']:+.1%} | mean tries-to-pass {mean_tries}")
        log(f"failed: {', '.join(failed) or '-'}")
    return run.run_id
