"""Metrics computed from stored results/attempts. Pure functions over the
dicts returned by ``storage`` so the CLI, dashboard and HTML report all show
identical numbers.

Headline metrics
  pass@1                 first-attempt pass rate (no feedback used)
  pass@k (k = 1..max)    cumulative pass rate if the loop stopped after k tries
  final pass rate        pass@max_tries, with a Wilson 95% interval
  self-correction lift   final - pass@1: what the feedback loop buys
  recovery rate          of tasks that failed try 1, the share later fixed
  tries-to-pass          distribution + mean among solved tasks
Loop-dynamics metrics
  marginal gain per try  pass@k - pass@(k-1): do retries plateau?
  error transitions      error class of try k -> try k+1 (or -> passed)
  partial credit         mean fraction of tests passed per attempt number
  retry outcomes         improved / same / regressed test-pass fraction
  stuck rate             retries that resubmitted identical code
  edit ratio             how much code changed between tries (targeted fix vs rewrite)
Cost metrics
  tokens, estimated USD, latency, tokens per solved task
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from agent_eval import config
from agent_eval.code_metrics import edit_ratio

TERMINAL_PASS = "passed"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (good at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = config.MODEL_PRICING.get(model, (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def group_attempts(attempts: list[dict]) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for a in attempts:
        by_task[a["task_id"]].append(a)
    for rows in by_task.values():
        rows.sort(key=lambda a: a["attempt_number"])
    return dict(by_task)


def first_pass_attempt(rows: list[dict]) -> int | None:
    for a in rows:
        if a["passed"]:
            return a["attempt_number"]
    return None


def _state(a: dict) -> str:
    return TERMINAL_PASS if a["passed"] else (a.get("error_type") or "wrong_answer")


def pass_at_k(results: list[dict], attempts: list[dict], max_tries: int) -> list[float]:
    by_task = group_attempts(attempts)
    n = len(results)
    curve = []
    for k in range(1, max_tries + 1):
        solved = 0
        for r in results:
            first = first_pass_attempt(by_task.get(r["task_id"], []))
            if first is not None and first <= k:
                solved += 1
        curve.append(solved / n if n else 0.0)
    return curve


def tries_distribution(results: list[dict], max_tries: int) -> dict[str, int]:
    dist = {str(k): 0 for k in range(1, max_tries + 1)}
    dist["failed"] = 0
    for r in results:
        if r["passed"]:
            dist[str(r["tries_taken"])] = dist.get(str(r["tries_taken"]), 0) + 1
        else:
            dist["failed"] += 1
    return dist


def error_transitions(attempts: list[dict]) -> Counter:
    """(state of try k, state of try k+1) counts across all tasks."""
    transitions: Counter = Counter()
    for rows in group_attempts(attempts).values():
        for prev, nxt in zip(rows, rows[1:], strict=False):
            transitions[(_state(prev), _state(nxt))] += 1
    return transitions


def retry_dynamics(attempts: list[dict]) -> dict:
    improved = same = regressed = stuck = 0
    edits: list[float] = []
    for rows in group_attempts(attempts).values():
        for prev, nxt in zip(rows, rows[1:], strict=False):
            before = prev["num_passed"] / prev["num_total"] if prev["num_total"] else 0.0
            after = nxt["num_passed"] / nxt["num_total"] if nxt["num_total"] else 0.0
            if nxt["passed"] and not prev["passed"]:
                after = 1.0
            if after > before:
                improved += 1
            elif after < before:
                regressed += 1
            else:
                same += 1
            if prev["code"].strip() == nxt["code"].strip():
                stuck += 1
            edits.append(edit_ratio(prev["code"], nxt["code"]))
    total = improved + same + regressed
    return {
        "retries": total,
        "improved": improved,
        "same": same,
        "regressed": regressed,
        "improved_rate": improved / total if total else None,
        "regression_rate": regressed / total if total else None,
        "stuck": stuck,
        "stuck_rate": stuck / total if total else None,
        "mean_edit_ratio": statistics.fmean(edits) if edits else None,
    }


def partial_credit_by_attempt(attempts: list[dict], max_tries: int) -> list[float | None]:
    """Mean fraction of tests passed at attempt k; a solved task counts as 1.0
    for every later k (carry-forward), an unsolved one carries its last score."""
    by_task = group_attempts(attempts)
    curve: list[float | None] = []
    for k in range(1, max_tries + 1):
        scores = []
        for rows in by_task.values():
            upto = [a for a in rows if a["attempt_number"] <= k]
            if not upto:
                continue
            last = upto[-1]
            if any(a["passed"] for a in upto):
                scores.append(1.0)
            else:
                scores.append(last["num_passed"] / last["num_total"] if last["num_total"] else 0.0)
        curve.append(statistics.fmean(scores) if scores else None)
    return curve


def by_difficulty(results: list[dict], attempts: list[dict], field: str = "difficulty",
                  order: dict[str, int] | None = None) -> dict[str, dict]:
    by_task = group_attempts(attempts)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r.get(field) or "unknown"].append(r)
    order = order or {"easy": 0, "medium": 1, "hard": 2}
    out = {}
    for diff in sorted(groups, key=lambda d: order.get(d, 9)):
        rows = groups[diff]
        n = len(rows)
        first = sum(1 for r in rows if first_pass_attempt(by_task.get(r["task_id"], [])) == 1)
        passed = sum(r["passed"] for r in rows)
        out[diff] = {"n": n, "pass_at_1": first / n, "pass_rate": passed / n,
                     "lift": (passed - first) / n}
    return out


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """ROC-AUC via the Mann-Whitney U statistic (average ranks for ties).
    None if one class is missing."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # 1-based average rank of the tie group
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def verifier_in_loop(attempts: list[dict]) -> dict | None:
    """How well the learned verifier's score on the code it picked predicted the
    sandbox verdict (only for best-of-N runs)."""
    scored = [a for a in attempts if a.get("verifier_score") is not None]
    if not scored:
        return None
    labels = [int(a["passed"]) for a in scored]
    scores = [a["verifier_score"] for a in scored]
    passed = [s for s, y in zip(scores, labels, strict=True) if y]
    failed = [s for s, y in zip(scores, labels, strict=True) if not y]
    return {
        "attempts": len(scored),
        "candidates": statistics.fmean(a.get("candidates") or 1 for a in scored),
        "auc": roc_auc(labels, scores),
        "mean_score_passed": statistics.fmean(passed) if passed else None,
        "mean_score_failed": statistics.fmean(failed) if failed else None,
    }


def summarize_run(run: dict, results: list[dict], attempts: list[dict]) -> dict:
    max_tries = int(run.get("max_tries") or max((r["tries_taken"] for r in results), default=1))
    n = len(results)
    n_passed = sum(r["passed"] for r in results)
    curve = pass_at_k(results, attempts, max_tries)
    p1 = curve[0] if curve else 0.0
    final = n_passed / n if n else 0.0
    by_task = group_attempts(attempts)
    failed_first = [r for r in results if first_pass_attempt(by_task.get(r["task_id"], [])) != 1]
    recovered = [r for r in failed_first if r["passed"]]
    tries_solved = [r["tries_taken"] for r in results if r["passed"]]

    failed_attempts = [a for a in attempts if not a["passed"]]
    error_counts = Counter(a.get("error_type") or "wrong_answer" for a in failed_attempts)
    first_errors = Counter(
        rows[0].get("error_type") or "wrong_answer"
        for rows in by_task.values() if rows and not rows[0]["passed"]
    )
    tokens_in = sum(a["tokens_in"] for a in attempts)
    tokens_out = sum(a["tokens_out"] for a in attempts)
    latencies = [r["latency_s"] for r in results]
    cost = estimate_cost(run.get("model", ""), tokens_in, tokens_out)

    quality = [r["quality_score"] for r in results if r.get("quality_score")]
    code = [r["code_metrics"] for r in results if r.get("code_metrics") and r["passed"]]

    lo, hi = wilson_interval(n_passed, n)
    return {
        "run_id": run.get("run_id"),
        "model": run.get("model"),
        "prompt_version": run.get("prompt_version"),
        "feedback_level": run.get("feedback_level"),
        "suite": run.get("suite", "easy"),
        "max_tries": max_tries,
        "seed": run.get("seed"),
        "timestamp": run.get("timestamp"),
        "num_tasks": n,
        "num_passed": n_passed,
        "pass_rate": final,
        "pass_rate_ci95": (lo, hi),
        "pass_at_1": p1,
        "pass_at_k": curve,
        "marginal_gain": [curve[0]] + [curve[i] - curve[i - 1] for i in range(1, len(curve))],
        "self_correction_lift": final - p1,
        "recovery_rate": len(recovered) / len(failed_first) if failed_first else None,
        "num_failed_first": len(failed_first),
        "num_recovered": len(recovered),
        "mean_tries_to_pass": statistics.fmean(tries_solved) if tries_solved else None,
        "tries_distribution": tries_distribution(results, max_tries),
        "total_attempts": len(attempts),
        "error_counts": dict(error_counts.most_common()),
        "first_attempt_errors": dict(first_errors.most_common()),
        "transitions": {f"{a} -> {b}": c for (a, b), c in error_transitions(attempts).most_common()},
        "retry_dynamics": retry_dynamics(attempts),
        "partial_credit": partial_credit_by_attempt(attempts, max_tries),
        "by_difficulty": by_difficulty(results, attempts),
        "by_category": by_difficulty(results, attempts, field="category", order={
            "function": 0, "bugfix": 1, "stateful": 2, "spec": 3, "performance": 4}),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_per_solved": (tokens_in + tokens_out) / n_passed if n_passed else None,
        "est_cost_usd": cost,
        "cost_per_solved_usd": cost / n_passed if n_passed else None,
        "latency_total_s": sum(latencies),
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "infra_errors": sum(1 for r in results if r.get("error")),
        "candidates": run.get("candidates") or 1,
        "verifier": run.get("verifier") or "",
        "verifier_in_loop": verifier_in_loop(attempts),
        "mean_quality": {
            key: statistics.fmean(q[key] for q in quality if key in q)
            for key in ("readability", "approach") if any(key in q for q in quality)
        } or None,
        "mean_code_metrics": {
            key: statistics.fmean(c[key] for c in code) for key in code[0]
        } if code else None,
    }


def aggregate_summaries(summaries: list[dict]) -> dict:
    """Mean and sample std-dev of headline metrics across repeated runs (seeds)."""
    keys = ("pass_rate", "pass_at_1", "self_correction_lift", "recovery_rate", "mean_tries_to_pass")
    out = {"num_runs": len(summaries)}
    for key in keys:
        values = [s[key] for s in summaries if s.get(key) is not None]
        out[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    curves = [s["pass_at_k"] for s in summaries if s.get("pass_at_k")]
    if curves:
        width = min(len(c) for c in curves)
        out["pass_at_k_mean"] = [statistics.fmean(c[i] for c in curves) for i in range(width)]
    return out


def load_summary(run_id: str) -> dict:
    from agent_eval import storage

    run = storage.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id!r}")
    return summarize_run(run, storage.get_results(run_id), storage.get_attempts(run_id))
