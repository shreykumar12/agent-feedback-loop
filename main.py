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


def _apply_rpm(args) -> None:
    if getattr(args, "rpm", None):
        config.LLM_RPM = args.rpm


def cmd_run(args) -> int:
    from agent_eval.agent import AgentError

    _apply_rpm(args)
    from agent_eval.run_suite import run_suite

    seeds = list(range(args.seed, args.seed + args.seeds))
    try:
        for seed in seeds:
            run_id = run_suite(
                model=args.model, prompt_version=args.prompt_version, task_ids=args.tasks,
                use_judge=args.judge, feedback_level=args.feedback_level, max_tries=args.max_tries,
                seed=seed, workers=args.workers, notes=args.notes, suite=args.suite,
                candidates=args.candidates, verifier=args.verifier, sample_temperature=args.sample_temperature,
            )
            print(run_id)
    except (AgentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _verifier_datasets(args):
    from agent_eval.ml import verifier as vf
    from agent_eval.tasks import load_tasks

    train_tasks = [t for s in args.train_suites for t in load_tasks(suite=s)]
    eval_tasks = [t for s in args.eval_suites for t in load_tasks(suite=s)]
    eval_ids = {t.task_id for t in eval_tasks}
    overlap = {t.task_id for t in train_tasks} & eval_ids
    if overlap:
        raise ValueError(f"train and eval suites share {len(overlap)} tasks; the test numbers would be meaningless")
    train = vf.examples_from_tasks(train_tasks)
    test = vf.examples_from_tasks(eval_tasks)
    if args.with_attempts:
        from agent_eval import storage

        storage.init_db()
        attempts = vf.examples_from_db()
        train += [e for e in attempts if e.task_id not in eval_ids]
        test += [e for e in attempts if e.task_id in eval_ids]
    train, val = vf.split_by_task(vf.dedupe(train), val_fraction=0.15, seed=args.seed)
    return train, val, vf.dedupe(test)


def cmd_train_verifier(args) -> int:
    from agent_eval.ml import verifier as vf

    try:
        train, val, test = _verifier_datasets(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"train {len(train)} examples ({sum(e.label for e in train)} pass) on "
          f"{len({e.task_id for e in train})} tasks | val {len(val)} | test {len(test)} on held-out suites "
          f"{' + '.join(args.eval_suites)}")
    model_cfg = vf.VerifierConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, d_ff=args.d_model * 3,
                                  canonical_names=args.canonical_names)
    verifier, result = vf.train_verifier(train, val, model_cfg, vf.TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed))
    test_metrics = vf.evaluate(verifier, test)
    print(f"\nbest epoch {result.best_epoch}")
    print("val :", vf.summarize(result.val))
    print("test:", vf.summarize(test_metrics))
    path = verifier.save(args.out, {"val": result.val, "test": test_metrics, "history": result.history,
                                    "train_suites": args.train_suites, "eval_suites": args.eval_suites})
    print(f"saved {path}")
    return 0


def cmd_eval_verifier(args) -> int:
    from agent_eval.ml import verifier as vf
    from agent_eval.tasks import load_tasks

    v = vf.load_verifier(args.path)
    for suite in args.suites:
        print(f"{suite:<6}", vf.summarize(vf.evaluate(v, vf.examples_from_tasks(load_tasks(suite=suite)))))
    return 0


def cmd_gen_train_tasks(args) -> int:
    from agent_eval.training_tasks import write_training_suite

    out = args.out or str(config.SUITES["train"])
    exclude = []
    for path in args.exclude or []:
        exclude += json.loads(open(path, encoding="utf-8").read())
    stats = write_training_suite(out, n=args.n, seed=args.seed, validate=not args.no_validate,
                                 id_prefix=args.id_prefix, exclude=exclude)
    print(f"wrote {stats['written']} tasks to {out} ({stats['dropped']} dropped by validation, "
          f"{stats['duplicates_removed']} duplicates of excluded tasks removed) across {len(stats['families'])} families")
    return 0


def cmd_build_suites(args) -> int:
    from agent_eval.suite_builder import build_suites

    stats = build_suites(config.SUITES["train"], config.SUITES["heldout"], pool_size=args.pool,
                         heldout_per_family=args.heldout_per_family, max_train=args.max_train,
                         seed=args.seed, validate=not args.no_validate)
    print(f"train {stats['train']} tasks -> {config.SUITES['train']}")
    print(f"heldout {stats['heldout']} tasks -> {config.SUITES['heldout']}")
    print(f"({stats['distinct']} distinct problems from a pool of {stats['pool']}, "
          f"{stats['dropped']} dropped by validation, {stats['families']} families)")
    if stats["families_without_heldout"]:
        print("families too small for a held-out task: " + ", ".join(stats["families_without_heldout"]))
    return 0


def cmd_export_sft(args) -> int:
    from agent_eval import storage
    from agent_eval.ml import sft

    run_ids = [rid for ref in args.runs for rid in storage.resolve_run_refs(ref)]
    examples = sft.examples_from_runs(run_ids, tuple(args.kinds))
    path = sft.write_jsonl(examples, args.out)
    print(f"{len(examples)} examples {sft.counts(examples)} from {len(run_ids)} run(s) -> {path}")
    return 0


def cmd_selftrain(args) -> int:
    from agent_eval.agent import AgentError
    from agent_eval.ml import finetune, selftrain

    lora = finetune.LoraTrainConfig(rank=args.lora_rank, alpha=args.lora_rank * 2, lr=args.lr, epochs=args.epochs,
                                    grad_accum=args.grad_accum, max_len=args.max_len, max_steps=args.max_steps)
    cfg = selftrain.SelfTrainConfig(
        base_model=args.model, experiment=args.experiment, rounds=args.rounds, train_suite=args.train_suite,
        eval_suites=tuple(args.eval_suites), train_tasks=args.train_tasks, eval_tasks=args.eval_tasks,
        feedback_level=args.feedback_level, max_tries=args.max_tries, prompt_version=args.prompt_version,
        warm_start=args.warm_start, seed=args.seed, lora=lora, kinds=tuple(args.kinds))
    try:
        rows = selftrain.self_train(cfg)
    except (ValueError, AgentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n" + selftrain.format_curve(rows))
    return 0


def cmd_compare_experiments(args) -> int:
    from agent_eval import regression
    from agent_eval.ml import selftrain

    results = selftrain.compare_experiments(args.treatment, args.control)
    if not results:
        print("the two experiments share no eval suites")
        return 2
    for suite, cmp in results.items():
        print(f"## {suite}: {args.treatment} (candidate) vs {args.control} (baseline), final rounds")
        print(regression.format_report(cmp))
    return 0


def cmd_curve(args) -> int:
    from agent_eval.ml import selftrain

    experiments = [args.experiment] if args.experiment else selftrain.list_experiments()
    if not experiments:
        print("no self-training experiments yet -- python main.py selftrain --model hf:<model> --experiment NAME")
        return 0
    for name in experiments:
        print(f"experiment {name}")
        print(selftrain.format_curve(selftrain.learning_curve(name)) + "\n")
    return 0


def cmd_runs(args) -> int:
    from agent_eval import storage

    storage.init_db()
    runs = storage.list_runs()[: args.limit]
    if not runs:
        print("no runs yet -- try: python main.py run --model sim-base")
        return 0
    header = f"{'run_id':<10}{'timestamp':<27}{'model':<20}{'suite':<7}{'prompt':<8}{'feedback':<10}{'tries':>6}{'seed':>6}{'tasks':>7}{'pass':>8}  status"
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['run_id'][:8]:<10}{r['timestamp']:<27}{r['model'][:19]:<20}{r.get('suite', 'easy'):<7}{r['prompt_version']:<8}"
              f"{r['feedback_level']:<10}{r['max_tries']:>6}{r['seed']:>6}{r['num_tasks']:>7}"
              f"{_pct(r['pass_rate']):>8}  {r['status']}")
    return 0


def _print_summary(s: dict) -> None:
    lo, hi = s["pass_rate_ci95"]
    print(f"run {s['run_id'][:8]}  {s['model']} / suite {s.get('suite', 'easy')} / prompt {s['prompt_version']} / feedback={s['feedback_level']}"
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
    if len(s["by_category"]) > 1:
        print("  by category         " + "  ".join(
            f"{c}: {v['pass_at_1']:.0%}->{v['pass_rate']:.0%} (n={v['n']})" for c, v in s["by_category"].items()))
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

    _apply_rpm(args)
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
                                    workers=args.workers, notes=tag, quiet=True, suite=args.suite)
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

    tasks = load_tasks(task_ids=args.tasks, suite=args.suite)
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


def _add_suite_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--suite", default=None,
                   help=f"task suite: {' / '.join(config.SUITES)} or a path to a tasks JSON (default: {config.DEFAULT_SUITE})")


def _add_loop_args(p: argparse.ArgumentParser) -> None:
    _add_suite_arg(p)
    p.add_argument("--prompt-version", default=config.DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--max-tries", type=int, default=config.MAX_TRIES)
    p.add_argument("--tasks", nargs="+", metavar="TASK_ID", help="only run these tasks")
    p.add_argument("--workers", type=int, default=None,
                   help="tasks run in parallel (default: 4 for sim-* models, 1 for real APIs)")
    p.add_argument("--rpm", type=float, default=None,
                   help="max LLM requests per minute, for rate-limited keys (e.g. 5 on a free tier)")
    p.add_argument("--notes", default="", help="free-text label stored with the run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="AgentEval: self-correcting coding agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite schema").set_defaults(func=cmd_init_db)

    p = sub.add_parser("run", help="run the suite under one configuration")
    p.add_argument("--model", default=config.GENERATOR_MODEL,
                   help="e.g. gemini-3.8-flash, or offline: sim-strong / sim-base / sim-weak")
    p.add_argument("--feedback-level", default=config.FEEDBACK_LEVEL, choices=FEEDBACK_LEVELS)
    p.add_argument("--judge", action="store_true", help="score passing code with the LLM judge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1, help="repeat with N consecutive seeds")
    p.add_argument("--candidates", type=int, default=1,
                   help="best-of-N: sample N programs per attempt, the --verifier picks which one is tested")
    p.add_argument("--verifier", default=None, help="path to a trained verifier (python main.py train-verifier)")
    p.add_argument("--sample-temperature", type=float, default=0.8,
                   help="temperature for candidates 2..N (candidate 1 uses the normal setting)")
    _add_loop_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("train-verifier", help="train the from-scratch PyTorch pass/fail verifier")
    p.add_argument("--train-suites", nargs="+", default=["train"],
                   help="suites whose reference solutions + mutants are training data")
    p.add_argument("--eval-suites", nargs="+", default=["easy", "hard"], help="held-out test suites")
    p.add_argument("--with-attempts", action="store_true",
                   help="also learn from every stored attempt (labeled by its sandbox verdict)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-canonical-names", dest="canonical_names", action="store_false",
                   help="keep original identifiers (default renames them to v0, v1, ... so the model "
                        "can't key on task-specific names)")
    p.add_argument("--out", default="models/verifier.pt")
    p.set_defaults(func=cmd_train_verifier)

    p = sub.add_parser("gen-train-tasks", help="(re)build the procedurally generated training suite")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: tasks/tasks_train.json")
    p.add_argument("--no-validate", action="store_true", help="skip running references + mutants in the sandbox")
    p.add_argument("--id-prefix", default="gen", help="task-id prefix; use a different one for a held-out set")
    p.add_argument("--exclude", nargs="+", metavar="JSON",
                   help="tasks files whose problems must not reappear (e.g. tasks/tasks_train.json)")
    p.set_defaults(func=cmd_gen_train_tasks)

    p = sub.add_parser("build-suites", help="build the train + held-out suites from one de-duplicated pool")
    p.add_argument("--pool", type=int, default=3000, help="candidate tasks to generate before de-duplication")
    p.add_argument("--heldout-per-family", type=int, default=3)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-validate", action="store_true")
    p.set_defaults(func=cmd_build_suites)

    p = sub.add_parser("export-sft", help="turn stored attempts into fine-tuning examples (JSONL)")
    p.add_argument("runs", nargs="+", help="run ids, prefixes, latest, comma lists or @model/... selectors")
    p.add_argument("--kinds", nargs="+", default=["direct", "repair", "distill"],
                   choices=["direct", "repair", "distill"])
    p.add_argument("--out", default="runs_ml/sft.jsonl")
    p.set_defaults(func=cmd_export_sft)

    p = sub.add_parser("selftrain", help="rounds of collect -> LoRA fine-tune -> eval on a local model")
    p.add_argument("--model", required=True, help="local base model, e.g. hf:Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--experiment", required=True, help="name for this run of rounds")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--train-suite", default="train")
    p.add_argument("--eval-suites", nargs="+", default=["heldout", "easy"],
                   help="held-out suites to measure each round on (hard is ~0%% for small models)")
    p.add_argument("--train-tasks", type=int, default=200, help="training tasks collected on per round")
    p.add_argument("--eval-tasks", type=int, default=None, help="limit eval tasks per suite (smoke tests)")
    p.add_argument("--feedback-level", default="full", choices=FEEDBACK_LEVELS)
    p.add_argument("--max-tries", type=int, default=config.MAX_TRIES)
    p.add_argument("--prompt-version", default=config.DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--warm-start", action="store_true",
                   help="also train on the training tasks' reference solutions (supervised bootstrap)")
    p.add_argument("--kinds", nargs="+", default=["direct", "repair", "distill"],
                   choices=["direct", "repair", "distill"],
                   help="which verified examples to train on; drop 'repair' for a no-feedback-learning control")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_selftrain)

    p = sub.add_parser("compare-experiments",
                       help="paired test of two self-training experiments' final models (e.g. with vs without feedback)")
    p.add_argument("treatment")
    p.add_argument("control")
    p.set_defaults(func=cmd_compare_experiments)

    p = sub.add_parser("curve", help="learning curve of a self-training experiment")
    p.add_argument("experiment", nargs="?")
    p.set_defaults(func=cmd_curve)

    p = sub.add_parser("eval-verifier", help="score a trained verifier on suites' references + mutants")
    p.add_argument("path")
    p.add_argument("--suites", nargs="+", default=["easy", "hard"])
    p.set_defaults(func=cmd_eval_verifier)

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
    _add_suite_arg(p)
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
