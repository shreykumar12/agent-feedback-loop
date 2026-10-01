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
    # First-try outcomes are paired too, and are far more sensitive: the retry
    # loop can rescue a weaker model's final pass rate while pass@1 collapses.
    base_by_task, cand_by_task = metrics.group_attempts(base_att), metrics.group_attempts(cand_att)
    base_first = {t: metrics.first_pass_attempt(base_by_task.get(t, [])) == 1 for t in common}
    cand_first = {t: metrics.first_pass_attempt(cand_by_task.get(t, [])) == 1 for t in common}
    first_try_lost = [t for t in common if base_first[t] and not cand_first[t]]
    first_try_gained = [t for t in common if not base_first[t] and cand_first[t]]

    p_value = mcnemar_exact(len(newly_failing), len(newly_passing))
    p_value_at_1 = mcnemar_exact(len(first_try_lost), len(first_try_gained))
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
        for key in ("model", "prompt_version", "feedback_level", "max_tries", "seed", "suite",
                    "candidates", "verifier")
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
        "first_try_lost": first_try_lost,
        "first_try_gained": first_try_gained,
        "mcnemar_p": p_value,
        "mcnemar_p_at_1": p_value_at_1,
        "pass_rate_delta_ci95": ci,
        "significant": p_value < SIGNIFICANCE_LEVEL,
        "significant_at_1": p_value_at_1 < SIGNIFICANCE_LEVEL,
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
        f"McNemar exact p (final pass): {comparison['mcnemar_p']:.3f} "
        f"({'significant' if comparison['significant'] else 'not significant'} at {SIGNIFICANCE_LEVEL})",
        f"McNemar exact p (pass@1):     {comparison['mcnemar_p_at_1']:.3f} "
        f"({'significant' if comparison['significant_at_1'] else 'not significant'}; "
        f"first-try lost {len(comparison['first_try_lost'])}, gained {len(comparison['first_try_gained'])})",
        "",
        f"newly failing ({len(comparison['newly_failing'])}): {', '.join(comparison['newly_failing']) or '-'}",
        f"newly passing ({len(comparison['newly_passing'])}): {', '.join(comparison['newly_passing']) or '-'}",
        "more tries: " + (", ".join(f"{m['task_id']} ({m['baseline']}->{m['candidate']})" for m in comparison["more_tries"]) or "-"),
        "fewer tries: " + (", ".join(f"{m['task_id']} ({m['baseline']}->{m['candidate']})" for m in comparison["fewer_tries"]) or "-"),
        "",
    ]
    if comparison["regression"]:
        lines.append("VERDICT: REGRESSION  -- " + "; ".join(comparison["reasons"]))
        if not (comparison["significant"] or comparison["significant_at_1"]):
            lines.append("         (not statistically significant: confirm with repeated runs, e.g.")
            lines.append("          `run --seeds 5` then `compare @model/feedback @model/feedback`)")
    else:
        lines.append("VERDICT: no regression")
    lines.append("=" * 64)
    return "\n".join(lines)


# --- repeated-run (group) comparison -----------------------------------------
# Single-run comparisons on a 20-task suite are noisy: an A/A comparison of the
# same configuration with two seeds can cross the thresholds above. Comparing
# groups of seeds with an exact permutation test separates real regressions
# from sampling noise.

def permutation_test(a: list[float], b: list[float], max_exact: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for mean(b) - mean(a)."""
    from itertools import combinations

    if not a or not b:
        return 1.0
    pooled = a + b
    n_a = len(a)
    observed = abs(sum(b) / len(b) - sum(a) / n_a)
    total = sum(pooled)

    def stat(idx_a) -> float:
        sum_a = sum(pooled[i] for i in idx_a)
        return abs((total - sum_a) / len(b) - sum_a / n_a)

    eps = 1e-12
    if math.comb(len(pooled), n_a) <= max_exact:
        splits = list(combinations(range(len(pooled)), n_a))
        hits = sum(1 for idx in splits if stat(idx) >= observed - eps)
        return hits / len(splits)
    rng = random.Random(seed)
    samples = max_exact
    hits = sum(1 for _ in range(samples) if stat(rng.sample(range(len(pooled)), n_a)) >= observed - eps)
    return (hits + 1) / (samples + 1)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compare_groups(baseline_ids: list[str], candidate_ids: list[str]) -> dict:
    groups = {}
    for name, ids in (("baseline", baseline_ids), ("candidate", candidate_ids)):
        runs, summaries, per_task_pass, per_task_first, by_seed = [], [], {}, {}, {}
        for rid in ids:
            run = storage.get_run(rid)
            if run is None:
                raise KeyError(f"unknown run {rid!r}")
            results, attempts = storage.get_results(rid), storage.get_attempts(rid)
            runs.append(run)
            summaries.append(metrics.summarize_run(run, results, attempts))
            by_task = metrics.group_attempts(attempts)
            for r in results:
                first = int(metrics.first_pass_attempt(by_task.get(r["task_id"], [])) == 1)
                per_task_pass.setdefault(r["task_id"], []).append(int(r["passed"]))
                per_task_first.setdefault(r["task_id"], []).append(first)
                by_seed[(run["seed"], r["task_id"])] = (int(r["passed"]), first)
        groups[name] = {"runs": runs, "summaries": summaries, "pass": per_task_pass,
                        "first": per_task_first, "by_seed": by_seed}

    b, c = groups["baseline"], groups["candidate"]
    common = sorted(b["pass"].keys() & c["pass"].keys())
    out_metrics = {}
    for key in ("pass_rate", "pass_at_1", "self_correction_lift", "recovery_rate", "mean_tries_to_pass"):
        xs = [s[key] for s in b["summaries"] if s[key] is not None]
        ys = [s[key] for s in c["summaries"] if s[key] is not None]
        out_metrics[key] = {
            "baseline_mean": _mean(xs), "baseline_std": _std(xs),
            "candidate_mean": _mean(ys), "candidate_std": _std(ys),
            "delta": (_mean(ys) - _mean(xs)) if xs and ys else None,
            "p_value": permutation_test(xs, ys),
        }

    degraded, improved = [], []
    for t in common:
        rate_b, rate_c = _mean(b["pass"][t]), _mean(c["pass"][t])
        first_b, first_c = _mean(b["first"][t]), _mean(c["first"][t])
        row = {"task_id": t, "baseline_solve": rate_b, "candidate_solve": rate_c,
               "baseline_first": first_b, "candidate_first": first_c}
        if rate_c - rate_b <= -0.5 or first_c - first_b <= -0.5:
            degraded.append(row)
        elif rate_c - rate_b >= 0.5 or first_c - first_b >= 0.5:
            improved.append(row)

    # Same seeds on both sides -> outcomes pair up by (seed, task): a pooled
    # McNemar test over all pairs is far more powerful than comparing run means.
    paired = None
    seeds_b = sorted(r["seed"] for r in b["runs"])
    seeds_c = sorted(r["seed"] for r in c["runs"])
    if seeds_b == seeds_c and len(set(seeds_b)) == len(seeds_b):
        keys = sorted(b["by_seed"].keys() & c["by_seed"].keys())
        paired = {"num_pairs": len(keys)}
        for idx, key in ((0, "pass_rate"), (1, "pass_at_1")):
            lost = sum(1 for k in keys if b["by_seed"][k][idx] and not c["by_seed"][k][idx])
            gained = sum(1 for k in keys if not b["by_seed"][k][idx] and c["by_seed"][k][idx])
            paired[key] = {"lost": lost, "gained": gained, "p_value": mcnemar_exact(lost, gained)}
            out_metrics[key]["paired_p_value"] = paired[key]["p_value"]

    reasons = []
    for key, threshold in (("pass_rate", PASS_RATE_DROP_THRESHOLD), ("pass_at_1", PASS_AT_1_DROP_THRESHOLD)):
        m = out_metrics[key]
        p_used = m.get("paired_p_value", m["p_value"])
        test = "paired McNemar" if "paired_p_value" in m else "permutation"
        if m["delta"] is not None and m["delta"] < -threshold and p_used < SIGNIFICANCE_LEVEL:
            reasons.append(f"{key} dropped {-m['delta']:.1%} (> {threshold:.0%}, {test} p={p_used:.3g})")

    warnings = []
    for name in ("baseline", "candidate"):
        runs = groups[name]["runs"]
        configs = {(r["model"], r["prompt_version"], r["feedback_level"], r["max_tries"], r.get("suite"),
                    r.get("candidates"), r.get("verifier")) for r in runs}
        if len(configs) > 1:
            warnings.append(f"{name} group mixes {len(configs)} configurations")
        if len({r["seed"] for r in runs}) < len(runs):
            warnings.append(f"{name} group repeats a seed (runs are not independent samples)")
    hashes = {r["suite_hash"] for g in groups.values() for r in g["runs"] if r.get("suite_hash")}
    if len(hashes) > 1:
        warnings.append("suite hash differs across runs: prompts or tests changed")
    if min(len(baseline_ids), len(candidate_ids)) < 3:
        warnings.append("fewer than 3 runs per group: the permutation test has little power")

    def describe(runs):
        r = runs[0]
        return {"model": r["model"], "prompt_version": r["prompt_version"], "feedback_level": r["feedback_level"],
                "max_tries": r["max_tries"], "run_ids": [x["run_id"] for x in runs],
                "seeds": sorted(x["seed"] for x in runs)}

    return {
        "baseline": describe(b["runs"]),
        "candidate": describe(c["runs"]),
        "num_common": len(common),
        "metrics": out_metrics,
        "paired": paired,
        "degraded_tasks": degraded,
        "improved_tasks": improved,
        "regression": bool(reasons),
        "reasons": reasons,
        "warnings": warnings,
    }


def format_group_report(comparison: dict) -> str:
    b, c = comparison["baseline"], comparison["candidate"]
    lines = [
        "=" * 72,
        "REGRESSION REPORT (repeated runs)",
        "=" * 72,
        f"baseline : {b['model']} / {b['prompt_version']} / fb={b['feedback_level']} / k={b['max_tries']}  ({len(b['run_ids'])} runs, seeds {b['seeds']})",
        f"candidate: {c['model']} / {c['prompt_version']} / fb={c['feedback_level']} / k={c['max_tries']}  ({len(c['run_ids'])} runs, seeds {c['seeds']})",
        f"tasks compared: {comparison['num_common']}",
    ]
    lines += [f"WARNING: {w}" for w in comparison["warnings"]]
    lines += ["", f"{'metric':<22}{'baseline':>18}{'candidate':>18}{'delta':>10}{'perm p':>9}"]
    for key, m in comparison["metrics"].items():
        is_pct = key != "mean_tries_to_pass"

        def cell(mean, std, pct=is_pct):
            if mean is None:
                return "-"
            return f"{mean:.1%} ± {std:.1%}" if pct else f"{mean:.2f} ± {std:.2f}"

        d = m["delta"]
        dtxt = "-" if d is None else (f"{d:+.1%}" if is_pct else f"{d:+.2f}")
        lines.append(f"{key:<22}{cell(m['baseline_mean'], m['baseline_std']):>18}"
                     f"{cell(m['candidate_mean'], m['candidate_std']):>18}{dtxt:>10}{m['p_value']:>9.3f}")
    if comparison.get("paired"):
        pr = comparison["paired"]
        lines.append("")
        lines.append(f"paired by (seed, task), {pr['num_pairs']} pairs:")
        for key in ("pass_rate", "pass_at_1"):
            lines.append(f"  {key:<12} lost {pr[key]['lost']:>3}, gained {pr[key]['gained']:>3}, "
                         f"McNemar p = {pr[key]['p_value']:.3g}")
    lines.append("")
    lines.append(f"degraded tasks ({len(comparison['degraded_tasks'])}): " + (", ".join(
        f"{t['task_id']} (solve {t['baseline_solve']:.0%}->{t['candidate_solve']:.0%}, "
        f"first-try {t['baseline_first']:.0%}->{t['candidate_first']:.0%})" for t in comparison["degraded_tasks"]) or "-"))
    lines.append(f"improved tasks ({len(comparison['improved_tasks'])}): " + (", ".join(
        t["task_id"] for t in comparison["improved_tasks"]) or "-"))
    lines.append("")
    if comparison["regression"]:
        lines.append("VERDICT: REGRESSION  -- " + "; ".join(comparison["reasons"]))
    else:
        lines.append("VERDICT: no statistically significant regression")
    lines.append("=" * 72)
    return "\n".join(lines)
