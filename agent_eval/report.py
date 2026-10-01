"""Self-contained HTML metrics report (``python main.py report``).

Charts are Vega-Lite specs rendered in the browser (vega/vega-lite/vega-embed
from jsdelivr); all data is embedded in the page as JSON, so a report is one
shareable file. Every chart has a data-table view, and colors come from CSS
tokens so light and dark themes both work.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from agent_eval import config, metrics, regression, storage
from agent_eval.feedback import FEEDBACK_LEVELS

TEMPLATE_PATH = Path(__file__).with_name("report_template.html")


def config_key(run: dict) -> str:
    suite = run.get("suite") or "easy"
    key = f"{run['model']} · {suite} · {run['prompt_version']} · fb={run['feedback_level']} · k={run['max_tries']}"
    if (run.get("candidates") or 1) > 1:
        key += f" · best-of-{run['candidates']}"
        if run.get("verifier"):
            key += f" ({Path(run['verifier']).stem})"
    return key


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _std(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_payload(run_ids: list[str] | None = None, focus_run_id: str | None = None,
                  compare: tuple[str, str] | None = None) -> dict:
    runs = storage.list_runs()
    if run_ids:
        wanted = set(run_ids)
        runs = [r for r in runs if r["run_id"] in wanted]
    runs = [r for r in runs if r["num_tasks"]]
    if not runs:
        raise KeyError("no completed runs to report on -- run `python main.py run --model sim-base` first")

    summaries, results_by_run = {}, {}
    for run in runs:
        results = storage.get_results(run["run_id"])
        attempts = storage.get_attempts(run["run_id"])
        results_by_run[run["run_id"]] = results
        summaries[run["run_id"]] = metrics.summarize_run(run, results, attempts)

    focus_id = focus_run_id or runs[0]["run_id"]  # runs are newest first
    focus = summaries[focus_id]

    # --- group repeated runs (seeds) of the same configuration --------------
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[config_key(run)].append(run)

    level_order = {lvl: i for i, lvl in enumerate(FEEDBACK_LEVELS)}
    configs, pass_at_k_rows, ablation_rows = [], [], []
    for key, members in groups.items():
        ss = [summaries[m["run_id"]] for m in members]
        first = members[0]
        agg = metrics.aggregate_summaries(ss)
        row = {
            "config": key, "model": first["model"], "prompt_version": first["prompt_version"],
            "feedback_level": first["feedback_level"], "max_tries": first["max_tries"],
            "n_runs": len(members), "num_tasks": ss[0]["num_tasks"],
            "pass_at_1": agg["pass_at_1"]["mean"], "pass_at_1_std": agg["pass_at_1"]["std"],
            "pass_rate": agg["pass_rate"]["mean"], "pass_rate_std": agg["pass_rate"]["std"],
            "lift": agg["self_correction_lift"]["mean"],
            "recovery_rate": agg["recovery_rate"]["mean"],
            "mean_tries": agg["mean_tries_to_pass"]["mean"],
            "tokens_per_solved": _mean([s["tokens_per_solved"] for s in ss]),
            "cost_usd": _mean([s["est_cost_usd"] for s in ss]),
            "regression_rate": _mean([s["retry_dynamics"]["regression_rate"] for s in ss]),
            "stuck_rate": _mean([s["retry_dynamics"]["stuck_rate"] for s in ss]),
        }
        configs.append(row)
        for k, value in enumerate(agg.get("pass_at_k_mean", []), 1):
            pass_at_k_rows.append({"config": key, "model": row["model"], "feedback_level": row["feedback_level"],
                                   "prompt_version": row["prompt_version"], "k": k, "pass_rate": value})
        ablation_rows.append({"model": row["model"], "prompt_version": row["prompt_version"],
                              "feedback_level": row["feedback_level"], "segment": "Solved on try 1",
                              "value": row["pass_at_1"], "final": row["pass_rate"],
                              "final_std": row["pass_rate_std"], "n_runs": row["n_runs"]})
        ablation_rows.append({"model": row["model"], "prompt_version": row["prompt_version"],
                              "feedback_level": row["feedback_level"], "segment": "Solved by a retry",
                              "value": row["lift"], "final": row["pass_rate"],
                              "final_std": row["pass_rate_std"], "n_runs": row["n_runs"]})
    configs.sort(key=lambda c: (c["model"], c["prompt_version"], level_order.get(c["feedback_level"], 9)))

    # --- per-task solve grid: task x configuration --------------------------
    grid = []
    for key, members in groups.items():
        per_task: dict[str, list[dict]] = defaultdict(list)
        for m in members:
            for r in results_by_run[m["run_id"]]:
                per_task[r["task_id"]].append(r)
        for task_id, rs in per_task.items():
            solved = [r for r in rs if r["passed"]]
            grid.append({
                "config": key, "task_id": task_id, "difficulty": rs[0].get("difficulty") or "medium",
                "solve_rate": len(solved) / len(rs), "solved": len(solved), "runs": len(rs),
                "mean_tries": _mean([r["tries_taken"] for r in solved]),
            })

    focus_run = next(r for r in runs if r["run_id"] == focus_id)
    tries_rows = [{"bucket": ("never passed" if b == "failed" else f"try {b}"), "count": c,
                   "passed": b != "failed"} for b, c in focus["tries_distribution"].items()]
    transitions = []
    for label, count in focus["transitions"].items():
        src, dst = label.split(" -> ")
        transitions.append({"from": src, "to": dst, "count": count})
    difficulty_rows = []
    for diff, d in focus["by_difficulty"].items():
        difficulty_rows.append({"difficulty": diff, "segment": "Solved on try 1", "value": d["pass_at_1"], "n": d["n"], "final": d["pass_rate"]})
        difficulty_rows.append({"difficulty": diff, "segment": "Solved by a retry", "value": d["lift"], "n": d["n"], "final": d["pass_rate"]})
    category_rows = []
    for cat, d in focus["by_category"].items():
        category_rows.append({"category": cat, "segment": "Solved on try 1", "value": d["pass_at_1"], "n": d["n"], "final": d["pass_rate"]})
        category_rows.append({"category": cat, "segment": "Solved by a retry", "value": d["lift"], "n": d["n"], "final": d["pass_rate"]})
    error_rows = [{"error_type": k, "count": v} for k, v in focus["error_counts"].items()]
    partial_rows = [{"k": i + 1, "credit": v} for i, v in enumerate(focus["partial_credit"]) if v is not None]

    comparison = None
    if compare:
        comparison = regression.compare_runs(*compare)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulated": any(r["model"].startswith("sim") for r in runs),
        "num_runs": len(runs),
        "focus": focus,
        "focus_run": {k: focus_run.get(k) for k in ("run_id", "model", "prompt_version", "feedback_level",
                                                    "max_tries", "seed", "timestamp", "suite_hash", "notes", "suite")},
        "configs": configs,
        "pass_at_k": pass_at_k_rows,
        "ablation": ablation_rows,
        "grid": grid,
        "tries": tries_rows,
        "transitions": transitions,
        "difficulty": difficulty_rows,
        "category": category_rows,
        "errors": error_rows,
        "partial": partial_rows,
        "comparison": comparison,
        "runs": [{k: r.get(k) for k in ("run_id", "timestamp", "model", "prompt_version", "feedback_level",
                                        "max_tries", "seed", "num_tasks", "pass_rate", "notes")} for r in runs],
    }


def render_html(payload: dict, standalone: bool = True) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data = json.dumps(payload, default=str).replace("</", "<\\/")  # keep </script> out of the JSON
    body = template.replace("__REPORT_DATA__", data)
    if not standalone:
        return body
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def write_report(run_ids: list[str] | None = None, out_path: str | Path | None = None,
                 focus_run_id: str | None = None, compare: tuple[str, str] | None = None,
                 standalone: bool = True) -> Path:
    payload = build_payload(run_ids, focus_run_id, compare)
    path = Path(out_path or config.REPORTS_DIR / "report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload, standalone=standalone), encoding="utf-8")
    return path

