"""Self-training rounds: the loop's own verified successes become training data.

    round 0:  evaluate the base model on the held-out eval suites
              collect: run the loop on TRAINING tasks, store every attempt
    round r:  fine-tune a fresh LoRA adapter on all verified examples so far
              (direct passes, repairs after feedback, distilled retries)
              evaluate the adapted model on the eval suites
              collect again with the adapted model (more and better data)

This is expert iteration / rejection-sampling fine-tuning: the sandbox is the
filter, so only code that passed hidden tests is ever a training target. The
eval suites are never used for training -- the curve measures generalization.

Each round is recorded in the ``training_rounds`` table so the learning curve
(pass@1, final pass rate, recovery per round and suite) can be shown by
``python main.py curve``, the report, and the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_eval import config, metrics, storage
from agent_eval.ml import finetune, local_model, sft

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_rounds (
    experiment      TEXT NOT NULL,
    round           INTEGER NOT NULL,
    model           TEXT NOT NULL,
    adapter         TEXT,
    examples        INTEGER NOT NULL DEFAULT 0,
    example_kinds   TEXT NOT NULL DEFAULT '{}',
    train_stats     TEXT NOT NULL DEFAULT '{}',
    collect_run_id  TEXT,
    eval_run_ids    TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    PRIMARY KEY (experiment, round)
);
"""


@dataclass
class SelfTrainConfig:
    base_model: str
    experiment: str
    rounds: int = 3
    train_suite: str = "train"
    eval_suites: tuple[str, ...] = ("heldout", "easy")
    train_tasks: int | None = 200  # how many training tasks to collect on per round
    eval_tasks: int | None = None  # None = whole eval suites
    feedback_level: str = "full"
    max_tries: int = 3
    prompt_version: str = "v1"
    warm_start: bool = False  # add training-task reference solutions to round-1 data
    kinds: tuple[str, ...] = sft.KINDS
    seed: int = 0
    out_dir: str = ""
    lora: finetune.LoraTrainConfig = field(default_factory=finetune.LoraTrainConfig)


def init_tables() -> None:
    storage.init_db()
    with storage.get_connection() as conn:
        conn.executescript(SCHEMA)
    conn.close()


def _record(experiment: str, round_: int, model: str, adapter: str | None, examples: list[dict],
            train_stats: dict, collect_run_id: str | None, eval_run_ids: dict) -> None:
    with storage.get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO training_rounds (experiment, round, model, adapter, examples, example_kinds,"
            " train_stats, collect_run_id, eval_run_ids, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment, round_, model, adapter, len(examples), json.dumps(sft.counts(examples)),
             json.dumps(train_stats), collect_run_id, json.dumps(eval_run_ids),
             datetime.now(UTC).isoformat(timespec="seconds")))
    conn.close()


def list_experiments() -> list[str]:
    init_tables()
    with storage.get_connection() as conn:
        rows = conn.execute("SELECT DISTINCT experiment FROM training_rounds ORDER BY experiment").fetchall()
    conn.close()
    return [r["experiment"] for r in rows]


def rounds(experiment: str) -> list[dict]:
    init_tables()
    with storage.get_connection() as conn:
        rows = conn.execute("SELECT * FROM training_rounds WHERE experiment = ? ORDER BY round",
                            (experiment,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("example_kinds", "train_stats", "eval_run_ids"):
            d[key] = json.loads(d[key])
        out.append(d)
    return out


def learning_curve(experiment: str) -> list[dict]:
    """One row per (round, suite): the eval metrics of that round's model, plus a
    paired comparison against round 0 on the same tasks (exact McNemar on
    first-try and final outcomes), so "it improved" comes with a p-value."""
    from agent_eval import regression

    rows = []
    base_runs: dict[str, str] = {}
    for r in rounds(experiment):
        for suite, run_id in r["eval_run_ids"].items():
            s = metrics.load_summary(run_id)
            base_runs.setdefault(suite, run_id)
            vs_base = {}
            if base_runs[suite] != run_id:
                cmp = regression.compare_runs(base_runs[suite], run_id)
                vs_base = {
                    "gained_at_1": len(cmp["first_try_gained"]), "lost_at_1": len(cmp["first_try_lost"]),
                    "p_at_1": cmp["mcnemar_p_at_1"], "gained": len(cmp["newly_passing"]),
                    "lost": len(cmp["newly_failing"]), "p": cmp["mcnemar_p"],
                }
            rows.append({**vs_base,
                "round": r["round"], "suite": suite, "run_id": run_id, "model": r["model"],
                "examples": r["examples"], "pass_at_1": s["pass_at_1"], "pass_rate": s["pass_rate"],
                "recovery_rate": s["recovery_rate"], "mean_tries_to_pass": s["mean_tries_to_pass"],
                "repair_examples": r["example_kinds"].get("repair", 0),
                "train_loss": r["train_stats"].get("final_loss"),
            })
    return rows


def self_train(cfg: SelfTrainConfig, log=print) -> list[dict]:
    from agent_eval.run_suite import run_suite
    from agent_eval.tasks import load_tasks

    if not local_model.is_local(cfg.base_model):
        raise ValueError("self-training needs a local model: --model hf:<repo-or-path>")
    init_tables()
    if rounds(cfg.experiment):
        raise ValueError(f"experiment {cfg.experiment!r} already exists; pick a new --experiment name")
    out_dir = Path(cfg.out_dir or config.PROJECT_ROOT / "adapters" / cfg.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    train_all = load_tasks(suite=cfg.train_suite)
    eval_ids = {t.task_id for s in cfg.eval_suites for t in load_tasks(suite=s)}
    if {t.task_id for t in train_all} & eval_ids:
        raise ValueError("training suite overlaps the eval suites; results would be contaminated")
    train_ids = [t.task_id for t in train_all][: cfg.train_tasks] if cfg.train_tasks else None

    def run(model: str, suite: str, task_ids, note: str, seed: int) -> str:
        return run_suite(model=model, prompt_version=cfg.prompt_version, suite=suite, task_ids=task_ids,
                         feedback_level=cfg.feedback_level, max_tries=cfg.max_tries, seed=seed,
                         notes=f"selftrain:{cfg.experiment}:{note}", quiet=True)

    def evaluate(model: str, round_: int) -> dict:
        ids = {}
        for suite in cfg.eval_suites:
            eval_ids_subset = [t.task_id for t in load_tasks(suite=suite)][: cfg.eval_tasks] if cfg.eval_tasks else None
            ids[suite] = run(model, suite, eval_ids_subset, f"r{round_}:eval", cfg.seed)
            s = metrics.load_summary(ids[suite])
            log(f"  round {round_} eval {suite:<5} pass@1 {s['pass_at_1']:.1%}  final {s['pass_rate']:.1%}")
        return ids

    model, adapter, collected, examples, stats = cfg.base_model, None, [], [], {}
    for round_ in range(cfg.rounds + 1):
        log(f"== round {round_}" + (" (base model)" if round_ == 0 else ""))
        if round_ > 0:
            examples = sft.examples_from_runs(collected, cfg.kinds)
            if cfg.warm_start:
                examples = sft.dedupe(examples + sft.examples_from_references(train_all, cfg.prompt_version))
            sft.write_jsonl(examples, out_dir / f"round_{round_}_data.jsonl")
            log(f"  training on {len(examples)} verified examples {sft.counts(examples)}")
            if not examples:
                log("  no verified successes to learn from yet -- try --warm-start or a stronger base model")
                _record(cfg.experiment, round_, model, adapter, examples, {}, None, {})
                break
            local_model.release()  # free the generation model before training on the same device
            adapter = str(out_dir / f"round_{round_}")
            stats = finetune.finetune_lora(local_model.parse_spec(cfg.base_model).base, examples, adapter,
                                           cfg.lora, log=log)
            log(f"  loss {stats['first_loss']:.3f} -> {stats['final_loss']:.3f} in {stats['seconds']}s")
            model = local_model.with_adapter(cfg.base_model, adapter)
            local_model.release()
        log(f"  evaluating {model}")
        eval_ids = evaluate(model, round_)
        collect_id = None
        if round_ < cfg.rounds:
            collect_id = run(model, cfg.train_suite, train_ids, f"r{round_}:collect", cfg.seed + round_)
            collected.append(collect_id)
            s = metrics.load_summary(collect_id)
            log(f"  round {round_} collect on {s['num_tasks']} training tasks: final {s['pass_rate']:.1%}")
        _record(cfg.experiment, round_, model, adapter, examples, stats, collect_id, eval_ids)
    return learning_curve(cfg.experiment)


def compare_experiments(treatment: str, control: str) -> dict[str, dict]:
    """Paired comparison of two experiments' FINAL-round models on each shared eval
    suite (e.g. trained with feedback vs. without). Same tasks, task by task."""
    from agent_eval import regression

    def final_runs(name: str) -> dict[str, str]:
        rs = rounds(name)
        if not rs:
            raise KeyError(f"unknown experiment {name!r}")
        return next((r["eval_run_ids"] for r in reversed(rs) if r["eval_run_ids"]), {})

    t, c = final_runs(treatment), final_runs(control)
    return {suite: regression.compare_runs(c[suite], t[suite]) for suite in sorted(t.keys() & c.keys())}


def format_curve(rows: list[dict]) -> str:
    if not rows:
        return "no rounds recorded"
    lines = [f"{'round':>5}  {'suite':<8}{'examples':>9}{'repairs':>8}{'pass@1':>9}{'final':>9}{'recovery':>10}"
             f"{'loss':>8}   vs round 0 (pass@1: +gained/-lost, McNemar p)"]
    for r in rows:
        def pct(v):
            return "-" if v is None else f"{v:.1%}"
        loss = "-" if r["train_loss"] is None else f"{r['train_loss']:.3f}"
        vs = ""
        if "p_at_1" in r:
            vs = f"   +{r['gained_at_1']}/-{r['lost_at_1']}  p={r['p_at_1']:.3g}"
        lines.append(f"{r['round']:>5}  {r['suite']:<8}{r['examples']:>9}{r['repair_examples']:>8}"
                     f"{pct(r['pass_at_1']):>9}{pct(r['pass_rate']):>9}{pct(r['recovery_rate']):>10}{loss:>8}{vs}")
    return "\n".join(lines)
