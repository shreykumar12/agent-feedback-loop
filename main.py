"""Command-line entry point. Parses args and calls the package -- no logic here.

    python main.py init-db
    python main.py run --model sim-base --prompt-version v1 [--feedback-level full] [--seeds 3]
    python main.py runs
    python main.py show latest
    python main.py compare <baseline> <candidate>      # exit 1 on regression
    python main.py compare @sim-strong/full @sim-weak/full   # all seeds, permutation test
    python main.py ablation --models sim-base --levels none minimal names full --seeds 3
    python main.py report latest --out reports/latest.html
    python main.py attempts latest <task_id>
    python main.py validate-tasks

Run ids accept a unique prefix, ``latest`` or ``latest~N``.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_eval import config
from agent_eval.feedback import FEEDBACK_LEVELS
from agent_eval.prompts import PROMPTS


def _pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def cmd_init_db(args) -> int:
    from agent_eval import storage

    storage.init_db()
    print(f"initialised {config.DB_PATH}")
    return 0


def cmd_run(args) -> int:
    from agent_eval.agent import AgentError
    from agent_eval.run_suite import run_suite

    seeds = list(range(args.seed, args.seed + args.seeds))
    try:
        for seed in seeds:
            run_id = run_suite(
                model=args.model, prompt_version=args.prompt_version, task_ids=args.tasks,
                use_judge=args.judge, feedback_level=args.feedback_level, max_tries=args.max_tries,
                seed=seed, workers=args.workers, notes=args.notes,
            )
            print(run_id)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_runs(args) -> int:
    from agent_eval import storage

    storage.init_db()
    runs = storage.list_runs()[: args.limit]
    if not runs:
        print("no runs yet -- try: python main.py run --model sim-base")
        return 0
    header = f"{'run_id':<10}{'timestamp':<27}{'model':<20}{'prompt':<8}{'feedback':<10}{'tries':>6}{'seed':>6}{'tasks':>7}{'pass':>8}  status"
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['run_id'][:8]:<10}{r['timestamp']:<27}{r['model'][:19]:<20}{r['prompt_version']:<8}"
              f"{r['feedback_level']:<10}{r['max_tries']:>6}{r['seed']:>6}{r['num_tasks']:>7}"
              f"{_pct(r['pass_rate']):>8}  {r['status']}")
    return 0


def _print_summary(s: dict) -> None:
    lo, hi = s["pass_rate_ci95"]
    print(f"run {s['run_id'][:8]}  {s['model']} / prompt {s['prompt_version']} / feedback={s['feedback_level']}"
          f" / max_tries={s['max_tries']} / seed={s['seed']}")
    print(f"  final pass rate     {_pct(s['pass_rate'])}  ({s['num_passed']}/{s['num_tasks']}, 95% CI {lo:.0%}-{hi:.0%})")
    print(f"  pass@1              {_pct(s['pass_at_1'])}")
    print("  pass@k              " + "  ".join(f"@{i + 1}:{v:.0%}" for i, v in enumerate(s["pass_at_k"])))
    print(f"  self-correction lift {s['self_correction_lift']:+.1%}")
    print(f"  recovery rate       {_pct(s['recovery_rate'])}  ({s['num_recovered']}/{s['num_failed_first']} first-try failures fixed)")
    mt = s["mean_tries_to_pass"]
    print(f"  mean tries-to-pass  {'-' if mt is None else f'{mt:.2f}'}")
    print("  tries distribution  " + "  ".join(f"{k}:{v}" for k, v in s["tries_distribution"].items()))
    print("  partial credit/try  " + "  ".join("-" if v is None else f"{v:.2f}" for v in s["partial_credit"]))
    rd = s["retry_dynamics"]
    if rd["retries"]:
        print(f"  retries             {rd['retries']}: improved {rd['improved']}, same {rd['same']},"
              f" regressed {rd['regressed']}, stuck (identical code) {rd['stuck']},"
              f" mean edit ratio {rd['mean_edit_ratio']:.2f}")
    print("  first-try errors    " + (", ".join(f"{k}={v}" for k, v in s["first_attempt_errors"].items()) or "-"))
    print("  all failed attempts " + (", ".join(f"{k}={v}" for k, v in s["error_counts"].items()) or "-"))
    print("  by difficulty       " + "  ".join(
        f"{d}: {v['pass_at_1']:.0%}->{v['pass_rate']:.0%} (n={v['n']})" for d, v in s["by_difficulty"].items()))
    print(f"  tokens              in {s['tokens_in']:,} / out {s['tokens_out']:,}"
          + (f" / {s['tokens_per_solved']:,.0f} per solved task" if s["tokens_per_solved"] else ""))
    print(f"  est. cost           ${s['est_cost_usd']:.4f}")
    print(f"  latency             total {s['latency_total_s']:.1f}s, p50 {s['latency_p50_s']:.2f}s, p95 {s['latency_p95_s']:.2f}s per task")
    if s["mean_code_metrics"]:
        cm = s["mean_code_metrics"]
        print(f"  code (passing)      sloc {cm['sloc']:.1f}, cyclomatic {cm['cyclomatic_complexity']:.1f}, nesting {cm['max_nesting']:.1f}")
    if s["mean_quality"]:
        print("  judge quality       " + ", ".join(f"{k} {v:.2f}/5" for k, v in s["mean_quality"].items()))
    if s["infra_errors"]:
        print(f"  infra errors        {s['infra_errors']} task(s) errored (see results.error)")


def cmd_show(args) -> int:
    from agent_eval import metrics, storage

    s = metrics.load_summary(storage.resolve_run_id(args.run))
    if args.json:
        print(json.dumps(s, indent=2, default=str))
    else:
        _print_summary(s)
    return 0


def cmd_compare(args) -> int:
    from agent_eval import regression, storage

    base_ids = storage.resolve_run_refs(args.baseline)
    cand_ids = storage.resolve_run_refs(args.candidate)
    if len(base_ids) == 1 and len(cand_ids) == 1:
        comparison = regression.compare_runs(base_ids[0], cand_ids[0])
        text = regression.format_report(comparison)
    else:
        comparison = regression.compare_groups(base_ids, cand_ids)
        text = regression.format_group_report(comparison)
    print(json.dumps(comparison, indent=2, default=str) if args.json else text)
    return 1 if comparison["regression"] else 0


def cmd_attempts(args) -> int:
    from agent_eval import storage

    run_id = storage.resolve_run_id(args.run)
    rows = storage.get_attempts(run_id, args.task_id)
    if not rows:
        print(f"no attempts for {args.task_id} in run {run_id[:8]}")
        return 1
    for a in rows:
        verdict = "PASS" if a["passed"] else (a["error_type"] or "FAIL")
        print(f"--- attempt {a['attempt_number']}: {verdict}  ({a['num_passed']}/{a['num_total']} tests)")
        if a["feedback_given"]:
            print("[feedback given]\n" + a["feedback_given"])
        print("[code]\n" + a["code"])
    return 0


def cmd_ablation(args) -> int:
    from agent_eval import metrics
    from agent_eval.agent import AgentError
    from agent_eval.run_suite import run_suite

    tag = args.notes or "ablation"
    rows = []
    try:
        for model in args.models:
            for level in args.levels:
                summaries = []
                for seed in range(args.seeds):
                    rid = run_suite(model=model, prompt_version=args.prompt_version, task_ids=args.tasks,
                                    feedback_level=level, max_tries=args.max_tries, seed=seed,
                                    workers=args.workers, notes=tag, quiet=True)
                    summaries.append(metrics.load_summary(rid))
                    print(f"  {model} fb={level} seed={seed}: {summaries[-1]['pass_rate']:.1%}", file=sys.stderr)
                rows.append((model, level, metrics.aggregate_summaries(summaries)))
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def cell(stat):
        if stat["mean"] is None:
            return "-"
        return f"{stat['mean']:.1%} ± {stat['std']:.1%}" if args.seeds > 1 else f"{stat['mean']:.1%}"

    print(f"\nFeedback-level ablation ({args.seeds} seed(s) per cell, prompt {args.prompt_version})")
    print(f"{'model':<14}{'feedback':<10}{'pass@1':>16}{'final':>16}{'lift':>16}{'recovery':>16}")
    for model, level, agg in rows:
        print(f"{model:<14}{level:<10}{cell(agg['pass_at_1']):>16}{cell(agg['pass_rate']):>16}"
              f"{cell(agg['self_correction_lift']):>16}{cell(agg['recovery_rate']):>16}")
    return 0


def cmd_report(args) -> int:
    from agent_eval import report, storage

    run_ids = [storage.resolve_run_id(r) for r in args.runs] if args.runs else None
    focus = storage.resolve_run_id(args.focus) if args.focus else None
    compare = tuple(storage.resolve_run_id(r) for r in args.compare) if args.compare else None
    path = report.write_report(run_ids=run_ids, out_path=args.out, focus_run_id=focus,
                               compare=compare, standalone=not args.fragment)
    print(path)
    return 0


def cmd_validate(args) -> int:
    from agent_eval.tasks import load_tasks, suite_hash, validate_suite

    tasks = load_tasks(task_ids=args.tasks)
    reports = validate_suite(tasks)
    print(f"{'task_id':<30}{'diff':<8}{'tests':>6}{'canonical':>11}{'mutants killed':>16}")
    for t, r in zip(tasks, reports, strict=False):
        canon = "-" if r.canonical_passed is None else ("pass" if r.canonical_passed else "FAIL")
        print(f"{r.task_id:<30}{t.difficulty:<8}{r.num_tests:>6}{canon:>11}{f'{r.mutants_killed}/{r.mutants_total}':>16}"
              + (f"  survivors: {r.surviving_mutants}" if r.surviving_mutants else ""))
        for failure in r.canonical_failures:
            print(f"    canonical failure: {failure[:120]}")
    total = sum(r.mutants_total for r in reports)
    killed = sum(r.mutants_killed for r in reports)
    print(f"\n{len(tasks)} tasks, {sum(r.num_tests for r in reports)} tests, suite hash {suite_hash(tasks)}")
    if total:
        print(f"mutation score: {killed}/{total} = {killed / total:.1%}")
    bad = [r.task_id for r in reports if not r.ok]
    if bad:
        print(f"INVALID: {', '.join(bad)}")
        return 1
    print("suite OK")
    return 0


def cmd_delete_run(args) -> int:
    from agent_eval import storage

    run_id = storage.resolve_run_id(args.run)
    storage.delete_run(run_id)
    print(f"deleted {run_id}")
    return 0


def _add_loop_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt-version", default=config.DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--max-tries", type=int, default=config.MAX_TRIES)
    p.add_argument("--tasks", nargs="+", metavar="TASK_ID", help="only run these tasks")
    p.add_argument("--workers", type=int, default=4, help="tasks run in parallel")
    p.add_argument("--notes", default="", help="free-text label stored with the run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="AgentEval: self-correcting coding agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite schema").set_defaults(func=cmd_init_db)

    p = sub.add_parser("run", help="run the suite under one configuration")
    p.add_argument("--model", default=config.GENERATOR_MODEL,
                   help="e.g. gemini-2.5-flash, or offline: sim-strong / sim-base / sim-weak")
    p.add_argument("--feedback-level", default=config.FEEDBACK_LEVEL, choices=FEEDBACK_LEVELS)
    p.add_argument("--judge", action="store_true", help="score passing code with the LLM judge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1, help="repeat with N consecutive seeds")
    _add_loop_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("runs", help="list stored runs")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="metrics summary for a run")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("compare", help="regression report (exit 1 if flagged)",
                       description="Each side is a run id/prefix, 'latest', a comma-separated list of runs, "
                                   "or a config selector @model[/feedback[/prompt]] (all its seeds). "
                                   "Multiple runs per side -> repeated-run comparison with a permutation test.")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("attempts", help="drill into one task's attempts")
    p.add_argument("run")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_attempts)

    p = sub.add_parser("ablation", help="grid of runs over models x feedback levels x seeds")
    p.add_argument("--models", nargs="+", default=["sim-base"])
    p.add_argument("--levels", nargs="+", default=list(FEEDBACK_LEVELS), choices=FEEDBACK_LEVELS)
    p.add_argument("--seeds", type=int, default=3)
    _add_loop_args(p)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("report", help="write a self-contained HTML metrics report")
    p.add_argument("runs", nargs="*", help="run ids (default: all runs)")
    p.add_argument("--out", default=None, help="output path (default reports/report.html)")
    p.add_argument("--focus", help="run whose detail charts are shown (default: newest)")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), help="include a regression panel")
    p.add_argument("--fragment", action="store_true", help="omit <html>/<head>/<body> (for embedding)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("validate-tasks", help="check canonical solutions + mutation score")
    p.add_argument("--tasks", nargs="+", metavar="TASK_ID")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("delete-run", help="remove a run and its attempts")
    p.add_argument("run")
    p.set_defaults(func=cmd_delete_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
