"""Turn the loop's own stored attempts into supervised fine-tuning examples.

Three kinds of example, each a chat prompt -> the code that passed:

  direct     first-attempt prompt -> code that passed on the first try
  repair     retry prompt (task + previous failing code + test feedback) -> the fix
             that then passed. This is what teaches the model to *use* feedback.
  distill    first-attempt prompt -> code that only passed after retries, so the
             next model can get it right without needing the retries.

Only sandbox-verified passing code becomes a target, so the data is exactly as
trustworthy as the hidden tests (whose quality the mutation score measures).
Prompts are rendered with the same prompt version and task category as the
run, so training matches what the model sees in the loop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_eval import metrics, prompts, storage

KINDS = ("direct", "repair", "distill")


def _target(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def examples_from_runs(run_ids: list[str], kinds: tuple[str, ...] = KINDS) -> list[dict]:
    out: list[dict] = []
    for run_id in run_ids:
        run = storage.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run {run_id!r}")
        version = run["prompt_version"]
        for task_id, rows in metrics.group_attempts(storage.get_attempts(run_id)).items():
            task = storage.get_task(task_id)
            if task is None:
                continue
            category = task.get("category") or "function"
            first_pass = metrics.first_pass_attempt(rows)
            if first_pass is None:
                continue
            passing = next(a for a in rows if a["attempt_number"] == first_pass)
            first_system, first_user = prompts.render(version, task["prompt"], category=category)
            if first_pass == 1 and "direct" in kinds:
                out.append({"kind": "direct", "task_id": task_id, "run_id": run_id,
                            "messages": _messages(first_system, first_user), "target": _target(passing["code"])})
            if first_pass > 1:
                prev = next(a for a in rows if a["attempt_number"] == first_pass - 1)
                if "repair" in kinds and passing.get("feedback_given"):
                    system, user = prompts.render(version, task["prompt"], previous_code=prev["code"],
                                                  feedback=passing["feedback_given"], category=category)
                    out.append({"kind": "repair", "task_id": task_id, "run_id": run_id,
                                "messages": _messages(system, user), "target": _target(passing["code"])})
                if "distill" in kinds:
                    out.append({"kind": "distill", "task_id": task_id, "run_id": run_id,
                                "messages": _messages(first_system, first_user), "target": _target(passing["code"])})
    return dedupe(out)


def examples_from_references(tasks, prompt_version: str = "v1") -> list[dict]:
    """Supervised warm start: training tasks' reference solutions as targets.
    Only for TRAINING suites; never pass eval-suite tasks here."""
    out = []
    for t in tasks:
        if t.canonical_solution:
            system, user = prompts.render(prompt_version, t.prompt, category=t.category)
            out.append({"kind": "reference", "task_id": t.task_id, "run_id": None,
                        "messages": _messages(system, user), "target": _target(t.canonical_solution)})
    return dedupe(out)


def dedupe(examples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for ex in examples:
        key = hashlib.sha1(json.dumps([ex["messages"], ex["target"]], sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def write_jsonl(examples: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def counts(examples: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        out[ex["kind"]] = out.get(ex["kind"], 0) + 1
    return out
