"""Build the training suite and its held-out eval suite from one de-duplicated pool.

Generating the two suites separately (say, with different seeds) lets the same
problem land in both, because a family's generator can draw the same variant
twice. Instead:

  1. generate a large pool of candidate tasks
  2. keep one task per distinct problem (``solution_fingerprint``: the reference
     solution with docstrings removed and the function renamed)
  3. drop anything that fails sandbox validation
  4. split every family: ``heldout_per_family`` tasks go to the held-out suite,
     the rest to training

So no problem appears twice in the training suite, none appears in both suites,
and the held-out suite covers every family the model trains on (in-distribution
generalization: same kinds of problems, never-seen instances).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from agent_eval.training_tasks import generate_training_tasks, solution_fingerprint, validate_generated


def _family(task: dict) -> str:
    return task["tags"][1]


def build_suites(train_path: str | Path, heldout_path: str | Path, pool_size: int = 3000,
                 heldout_per_family: int = 3, max_train: int | None = None, seed: int = 0,
                 validate: bool = True, log=print) -> dict:
    pool = generate_training_tasks(pool_size, seed=seed, id_prefix="pool")
    distinct, seen = [], set()
    for task in pool:
        fp = solution_fingerprint(task)
        if fp not in seen:
            seen.add(fp)
            distinct.append(task)
    log(f"pool {len(pool)} -> {len(distinct)} distinct problems")
    dropped = 0
    if validate:
        bad = {p.split(":", 1)[0] for p in validate_generated(distinct)}
        dropped = len(bad)
        distinct = [t for t in distinct if t["task_id"] not in bad]
        log(f"validation dropped {dropped}")

    by_family: dict[str, list[dict]] = {}
    for task in distinct:
        by_family.setdefault(_family(task), []).append(task)
    train, heldout = [], []
    for family in sorted(by_family):
        tasks = by_family[family]
        random.Random(f"split/{seed}/{family}").shuffle(tasks)
        k = min(heldout_per_family, max(0, len(tasks) - 1))  # keep >= 1 for training
        heldout += tasks[:k]
        train += tasks[k:]
    if max_train is not None:
        train = train[:max_train]

    def relabel(tasks: list[dict], prefix: str) -> list[dict]:
        counters: dict[str, int] = {}
        out = []
        for task in sorted(tasks, key=lambda t: (_family(t), t["task_id"])):
            family = _family(task)
            counters[family] = counters.get(family, 0) + 1
            out.append(dict(task, task_id=f"{prefix}_{family}_{counters[family]:04d}"))
        return out

    train, heldout = relabel(train, "gen"), relabel(heldout, "heldout")
    for path, tasks in ((train_path, train), (heldout_path, heldout)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "pool": len(pool), "distinct": len(seen), "dropped": dropped,
        "train": len(train), "heldout": len(heldout), "families": len(by_family),
        "families_without_heldout": sorted(f for f, t in by_family.items() if len(t) <= 1),
    }
