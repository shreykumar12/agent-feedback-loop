"""Answer "after changing the model or prompt, did things get worse?"

Compares two stored runs task-by-task (paired), flags regressions against
explicit thresholds, and attaches statistics so a flag can be weighed against
noise: with a 20-task suite a single flipped task moves the pass rate by 5
points, so an exact McNemar test on the discordant pairs and a paired
bootstrap interval on the pass-rate delta are reported alongside.
"""

from __future__ import annotations

import math
import random

from agent_eval import metrics, storage

# --- thresholds: a candidate is flagged as a regression if ANY holds ---------
MAX_NEWLY_FAILING = 0             # any task that passed in baseline but fails now
PASS_RATE_DROP_THRESHOLD = 0.05   # absolute drop in final pass rate
PASS_AT_1_DROP_THRESHOLD = 0.10   # absolute drop in first-try pass rate
MEAN_TRIES_INCREASE_THRESHOLD = 0.5
SIGNIFICANCE_LEVEL = 0.05
BOOTSTRAP_SAMPLES = 5000


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(base: list[int], cand: list[int], samples: int = BOOTSTRAP_SAMPLES,
                        seed: int = 0) -> tuple[float, float]:
    """95% CI for mean(cand) - mean(base), resampling tasks with replacement."""
    n = len(base)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs = [c - b for b, c in zip(base, cand, strict=False)]
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return (stats[int(0.025 * samples)], stats[int(0.975 * samples) - 1])


def compare_results(baseline_run: dict, candidate_run: dict,
                    base_results: list[dict], cand_results: list[dict],
                    base_attempts: list[dict], cand_attempts: list[dict]) -> dict:
    base = {r["task_id"]: r for r in base_results}
    cand = {r["task_id"]: r for r in cand_results}
    common = sorted(base.keys() & cand.keys())
    only_base = sorted(base.keys() - cand.keys())
    only_cand = sorted(cand.keys() - base.keys())

    base_common = [base[t] for t in common]
    cand_common = [cand[t] for t in common]
    base_att = [a for a in base_attempts if a["task_id"] in common]
    cand_att = [a for a in cand_attempts if a["task_id"] in common]
    s_base = metrics.summarize_run(baseline_run, base_common, base_att)
    s_cand = metrics.summarize_run(candidate_run, cand_common, cand_att)

    newly_failing = [t for t in common if base[t]["passed"] and not cand[t]["passed"]]
    newly_passing = [t for t in common if not base[t]["passed"] and cand[t]["passed"]]
    both = [t for t in common if base[t]["passed"] and cand[t]["passed"]]
    more_tries = [
        {"task_id": t, "baseline": base[t]["tries_taken"], "candidate": cand[t]["tries_taken"]}
        for t in both if cand[t]["tries_taken"] > base[t]["tries_taken"]
    ]
    fewer_tries = [
        {"task_id": t, "baseline": base[t]["tries_taken"], "candidate": cand[t]["tries_taken"]}
        for t in both if cand[t]["tries_taken"] < base[t]["tries_taken"]
    ]

    def delta(key):
        a, b = s_base.get(key), s_cand.get(key)
        return None if a is None or b is None else b - a

    aggregate = {
        key: {"baseline": s_base.get(key), "candidate": s_cand.get(key), "delta": delta(key)}
        for key in ("pass_rate", "pass_at_1", "self_correction_lift", "recovery_rate",
                    "mean_tries_to_pass", "tokens_in", "tokens_out", "est_cost_usd",
                    "latency_total_s")
    }
    p_value = mcnemar_exact(len(newly_failing), len(newly_passing))
    ci = paired_bootstrap_ci([int(base[t]["passed"]) for t in common],
                             [int(cand[t]["passed"]) for t in common])

    reasons = []
    if len(newly_failing) > MAX_NEWLY_FAILING:
        reasons.append(f"{len(newly_failing)} task(s) newly failing")
    if (d := delta("pass_rate")) is not None and d < -PASS_RATE_DROP_THRESHOLD:
        reasons.append(f"pass rate dropped {-d:.1%} (> {PASS_RATE_DROP_THRESHOLD:.0%})")
    if (d := delta("pass_at_1")) is not None and d < -PASS_AT_1_DROP_THRESHOLD:
        reasons.append(f"pass@1 dropped {-d:.1%} (> {PASS_AT_1_DROP_THRESHOLD:.0%})")
    if (d := delta("mean_tries_to_pass")) is not None and d > MEAN_TRIES_INCREASE_THRESHOLD:
        reasons.append(f"mean tries-to-pass rose by {d:.2f} (> {MEAN_TRIES_INCREASE_THRESHOLD})")

    warnings = []
    if only_base or only_cand:
        warnings.append(f"task sets differ: {len(only_base)} only in baseline, "
                        f"{len(only_cand)} only in candidate; compared {len(common)} common tasks")
    if baseline_run.get("suite_hash") and candidate_run.get("suite_hash") \
            and baseline_run["suite_hash"] != candidate_run["suite_hash"]:
        warnings.append("suite hash differs: prompts or tests changed between runs")
    if baseline_run.get("max_tries") != candidate_run.get("max_tries"):
        warnings.append("max_tries differs between runs")

    changed = {
        key: (baseline_run.get(key), candidate_run.get(key))
        for key in ("model", "prompt_version", "feedback_level", "max_tries", "seed")
        if baseline_run.get(key) != candidate_run.get(key)
    }
    return {
        "baseline": {k: baseline_run.get(k) for k in ("run_id", "model", "prompt_version", "feedback_level", "max_tries", "seed")},
        "candidate": {k: candidate_run.get(k) for k in ("run_id", "model", "prompt_version", "feedback_level", "max_tries", "seed")},
        "changed": changed,
        "num_common": len(common),
        "only_in_baseline": only_base,
        "only_in_candidate": only_cand,
        "aggregate": aggregate,
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "more_tries": more_tries,
        "fewer_tries": fewer_tries,
        "mcnemar_p": p_value,
        "pass_rate_delta_ci95": ci,
        "significant": p_value < SIGNIFICANCE_LEVEL,
        "regression": bool(reasons),
        "reasons": reasons,
        "warnings": warnings,
        "pass_at_k": {"baseline": s_base["pass_at_k"], "candidate": s_cand["pass_at_k"]},
    }


def compare_runs(baseline_run_id: str, candidate_run_id: str) -> dict:
    runs = {}
    for rid in (baseline_run_id, candidate_run_id):
        run = storage.get_run(rid)
        if run is None:
            raise KeyError(f"unknown run {rid!r}")
        runs[rid] = run
    return compare_results(
        runs[baseline_run_id], runs[candidate_run_id],
        storage.get_results(baseline_run_id), storage.get_results(candidate_run_id),
        storage.get_attempts(baseline_run_id), storage.get_attempts(candidate_run_id),
    )


def _fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "dpct":
        return f"{value:+.1%}"
    if kind == "usd":
        return f"${value:.4f}"
    if kind == "dusd":
        return f"{value:+.4f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "dint":
        return f"{value:+,.0f}"
    if kind == "dfloat":
        return f"{value:+.2f}"
    return f"{value:.2f}"


def format_report(comparison: dict) -> str:
    b, c = comparison["baseline"], comparison["candidate"]
    lines = [
        "=" * 64,
        "REGRESSION REPORT",
        "=" * 64,
        f"baseline : {b['run_id'][:8]}  {b['model']} / {b['prompt_version']} / fb={b['feedback_level']} / seed={b['seed']}",
        f"candidate: {c['run_id'][:8]}  {c['model']} / {c['prompt_version']} / fb={c['feedback_level']} / seed={c['seed']}",
    ]
    if comparison["changed"]:
        lines.append("changed  : " + ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in comparison["changed"].items()))
    lines.append(f"tasks compared: {comparison['num_common']}")
    for w in comparison["warnings"]:
        lines.append(f"WARNING: {w}")

    lines += ["", f"{'metric':<22}{'baseline':>12}{'candidate':>12}{'delta':>12}"]
    kinds = {
        "pass_rate": ("pct", "dpct"), "pass_at_1": ("pct", "dpct"),
        "self_correction_lift": ("pct", "dpct"), "recovery_rate": ("pct", "dpct"),
        "mean_tries_to_pass": ("float", "dfloat"), "tokens_in": ("int", "dint"),
        "tokens_out": ("int", "dint"), "est_cost_usd": ("usd", "dusd"),
        "latency_total_s": ("float", "dfloat"),
    }
    for key, row in comparison["aggregate"].items():
        k, dk = kinds[key]
        lines.append(f"{key:<22}{_fmt(row['baseline'], k):>12}{_fmt(row['candidate'], k):>12}{_fmt(row['delta'], dk):>12}")

    lo, hi = comparison["pass_rate_delta_ci95"]
    lines += [
        "",
        f"pass-rate delta 95% CI (paired bootstrap): [{lo:+.1%}, {hi:+.1%}]",
        f"McNemar exact p-value: {comparison['mcnemar_p']:.3f} "
        f"({'significant' if comparison['significant'] else 'not significant'} at {SIGNIFICANCE_LEVEL})",
        "",
        f"newly failing ({len(comparison['newly_failing'])}): {', '.join(comparison['newly_failing']) or '-'}",
        f"newly passing ({len(comparison['newly_passing'])}): {', '.join(comparison['newly_passing']) or '-'}",
        "more tries: " + (", ".join(f"{m['task_id']} ({m['baseline']}->{m['candidate']})" for m in comparison["more_tries"]) or "-"),
        "fewer tries: " + (", ".join(f"{m['task_id']} ({m['baseline']}->{m['candidate']})" for m in comparison["fewer_tries"]) or "-"),
        "",
    ]
    if comparison["regression"]:
        lines.append("VERDICT: REGRESSION  -- " + "; ".join(comparison["reasons"]))
        if not comparison["significant"]:
            lines.append("         (not statistically significant: consider repeated runs with --seeds)")
    else:
        lines.append("VERDICT: no regression")
    lines.append("=" * 64)
    return "\n".join(lines)
