"""PyTorch side: local models through LangChain, SFT export, LoRA, the
from-scratch verifier, best-of-N, and self-training rounds.

Uses a tiny randomly initialized Llama built on the fly, so it needs no
download and runs on CPU. Skipped entirely if the ML extras aren't installed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("langchain_huggingface")

from agent_eval import storage  # noqa: E402
from agent_eval.graph import LoopSettings, run_task  # noqa: E402
from agent_eval.ml import finetune, local_model, sft  # noqa: E402
from agent_eval.ml import verifier as vf  # noqa: E402
from agent_eval.ml.tiny import build_tiny_model  # noqa: E402
from agent_eval.run_suite import run_suite  # noqa: E402
from agent_eval.tasks import load_tasks  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = build_tiny_model(tmp_path_factory.mktemp("tiny") / "model")
    yield f"hf:{path}"
    local_model.release()


@pytest.fixture(autouse=True)
def short_generations(monkeypatch):
    monkeypatch.setattr(local_model, "MAX_NEW_TOKENS", 16)


# --- local models -----------------------------------------------------------

def test_spec_parsing():
    spec = local_model.parse_spec("hf:Qwen/Qwen2.5-Coder-0.5B-Instruct+adapters/x/round_1")
    assert spec.base == "Qwen/Qwen2.5-Coder-0.5B-Instruct" and spec.adapter == "adapters/x/round_1"
    assert str(spec) == "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct+adapters/x/round_1"
    assert local_model.with_adapter(str(spec), None) == "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct"
    with pytest.raises(ValueError):
        local_model.parse_spec("gemini-3.8-flash")


def test_local_model_runs_through_langchain_and_the_loop(tiny_model, tmp_db):
    text, tokens_in, tokens_out = local_model.load(tiny_model).invoke("system", "def f(x):", max_new_tokens=8)
    assert isinstance(text, str) and tokens_in > 0
    run_id = run_suite(model=tiny_model, prompt_version="v1", task_ids=["clamp_all"], max_tries=2, quiet=True)
    attempts = storage.get_attempts(run_id)
    assert len(attempts) == 2  # the nonsense model fails, so the loop retries
    assert all(a["tokens_in"] > 0 for a in attempts)
    assert attempts[1]["feedback_given"]  # feedback reached the local model too


# --- SFT export ---------------------------------------------------------------

def test_sft_examples_include_repairs_with_feedback(tmp_db):
    run_ids = [run_suite(model="sim-weak", prompt_version="v1", seed=s, quiet=True,
                         task_ids=["rounded_mean", "merge_intervals", "rank_players", "window_maxima"])
               for s in range(4)]
    examples = sft.examples_from_runs(run_ids)
    kinds = sft.counts(examples)
    assert kinds.get("repair", 0) > 0 and kinds.get("distill", 0) > 0
    repair = next(e for e in examples if e["kind"] == "repair")
    assert "Your previous attempt" in repair["messages"][1]["content"]
    assert "tests passed" in repair["messages"][1]["content"] or "test" in repair["messages"][1]["content"]
    assert repair["target"].startswith("```python\n")
    # targets are only ever sandbox-verified passing code
    passing = {a["code"].strip() for r in run_ids for a in storage.get_attempts(r) if a["passed"]}
    assert all(e["target"][len("```python\n"):-len("\n```")] in passing for e in examples)


def test_reference_examples_and_jsonl_roundtrip(tmp_path):
    examples = sft.examples_from_references(load_tasks(suite="easy")[:3])
    assert len(examples) == 3 and {e["kind"] for e in examples} == {"reference"}
    path = sft.write_jsonl(examples, tmp_path / "d.jsonl")
    assert sft.read_jsonl(path) == examples


# --- LoRA -----------------------------------------------------------------------

def test_lora_finetune_lowers_loss_and_adapter_loads(tiny_model, tmp_path):
    base = local_model.parse_spec(tiny_model).base
    examples = sft.examples_from_references(load_tasks(suite="easy")[:6]) * 3
    cfg = finetune.LoraTrainConfig(rank=8, alpha=16, lr=5e-3, epochs=3, grad_accum=1, max_len=4096,
                                   gradient_checkpointing=False, eval_fraction=0.0)
    stats = finetune.finetune_lora(base, examples, tmp_path / "adapter", cfg, log=lambda *a: None)
    assert stats["final_loss"] < stats["first_loss"]
    assert 0 < stats["trainable_params"] < stats["total_params"]
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    adapted = local_model.load(local_model.with_adapter(tiny_model, str(tmp_path / "adapter")))
    assert isinstance(adapted.invoke("s", "u", max_new_tokens=4)[0], str)


# --- verifier -------------------------------------------------------------------

def test_auc_and_selection_metrics():
    assert vf.roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert vf.roc_auc([1, 0], [0.5, 0.5]) == 0.5
    assert vf.roc_auc([1, 1], [0.1, 0.2]) is None
    ex = [vf.Example("t", "p", "a", 1), vf.Example("t", "p", "b", 0), vf.Example("t", "p", "c", 0)]
    tied = vf.selection_metrics(ex, [0.5, 0.5, 0.5])
    assert tied["top1_pass_rate"] == pytest.approx(1 / 3)  # ties never broken by the label
    assert vf.selection_metrics(ex, [0.9, 0.1, 0.2])["top1_pass_rate"] == 1.0


def test_verifier_learns_a_separable_signal_and_roundtrips(tmp_path):
    # Passing programs return x; failing ones contain an obvious bug marker.
    def make(i, ok):
        body = "    return x\n" if ok else "    return x + 1  # off by one\n"
        return vf.Example(f"t{i}", f"def f{i}(x): ...", f"def f{i}(x):\n{body}", int(ok))
    data = [make(i, ok) for i in range(60) for ok in (True, False)]
    train, val = vf.split_by_task(data, val_fraction=0.25, seed=0)
    assert not {e.task_id for e in train} & {e.task_id for e in val}  # task-level split
    model_cfg = vf.VerifierConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, max_len=128, prompt_budget=32)
    verifier, result = vf.train_verifier(train, val, model_cfg, vf.TrainConfig(epochs=12, batch_size=16, lr=3e-3),
                                         log=lambda *a: None)
    assert result.val["auc"] > 0.9
    path = verifier.save(tmp_path / "v.pt", {"val": result.val})
    loaded = vf.load_verifier(path, device="cpu")
    pair = (val[0].prompt, val[0].code)
    assert loaded.score(*pair) == pytest.approx(verifier.score(*pair), abs=1e-5)
    assert vf.checkpoint_metrics(path)["val"]["auc"] == result.val["auc"]


def test_byte_patching_handles_odd_lengths():
    net = vf.VerifierNet(vf.VerifierConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, max_len=64, prompt_budget=8))
    ids, mask = vf.ByteTokenizer(64, 8).batch([("p", "abc"), ("prompt", "x" * 40)])
    assert net(ids, mask).shape == (2,)


# --- best-of-N ------------------------------------------------------------------

class OracleVerifier:
    """Scores the reference solution highest: an upper bound on reranking."""

    def __init__(self, tasks):
        self.good = {t.canonical_solution.strip() for t in tasks}

    def score_batch(self, pairs):
        return [1.0 if code.strip() in self.good else 0.0 for _, code in pairs]


def test_best_of_n_improves_pass_at_1_and_is_paired(monkeypatch):
    tasks = load_tasks(suite="easy")
    monkeypatch.setattr(vf, "load_verifier_cached", lambda path: OracleVerifier(tasks))
    single = [run_task(t, LoopSettings(model="sim-weak", max_tries=1, seed=3))[0] for t in tasks]
    best = [run_task(t, LoopSettings(model="sim-weak", max_tries=1, seed=3, candidates=5,
                                     verifier="oracle"))[0] for t in tasks]
    assert sum(r.passed for r in best) > sum(r.passed for r in single)
    # candidate 0 is the single-sample output: whatever passed alone still passes
    assert all(b.passed for s, b in zip(single, best, strict=True) if s.passed)
    # the cost of every candidate is charged to the attempt
    assert sum(r.total_tokens_out for r in best) > sum(r.total_tokens_out for r in single)


def test_best_of_n_records_candidates_and_score(monkeypatch):
    task = load_tasks(suite="easy")[0]
    monkeypatch.setattr(vf, "load_verifier_cached", lambda path: OracleVerifier([task]))
    _, attempts = run_task(task, LoopSettings(model="sim-base", max_tries=1, candidates=3, verifier="x"))
    assert attempts[0].candidates == 3 and attempts[0].verifier_score is not None


def test_best_of_n_requires_a_verifier(tmp_db):
    with pytest.raises(ValueError, match="verifier"):
        run_suite(model="sim-base", prompt_version="v1", task_ids=["clamp_all"], candidates=3, quiet=True)


# --- self-training rounds --------------------------------------------------------

def test_selftrain_records_a_learning_curve(tiny_model, tmp_db, tmp_path):
    from agent_eval.ml import selftrain

    cfg = selftrain.SelfTrainConfig(
        base_model=tiny_model, experiment="t", rounds=1, train_suite="hard", eval_suites=("easy",),
        train_tasks=2, eval_tasks=2, max_tries=1, warm_start=True, out_dir=str(tmp_path / "adapters"),
        lora=finetune.LoraTrainConfig(rank=4, alpha=8, max_steps=1, grad_accum=1, max_len=4096,
                                      gradient_checkpointing=False))
    rows = selftrain.self_train(cfg, log=lambda *a: None)
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[1]["examples"] > 0 and rows[1]["train_loss"] is not None
    assert (tmp_path / "adapters" / "round_1" / "adapter_config.json").exists()
    assert "round" in selftrain.format_curve(rows)
    with pytest.raises(ValueError, match="already exists"):
        selftrain.self_train(cfg, log=lambda *a: None)


def test_selftrain_refuses_contaminated_suites(tiny_model, tmp_db):
    from agent_eval.ml import selftrain

    cfg = selftrain.SelfTrainConfig(base_model=tiny_model, experiment="bad", train_suite="easy",
                                    eval_suites=("easy",))
    with pytest.raises(ValueError, match="overlaps"):
        selftrain.self_train(cfg, log=lambda *a: None)


def test_selftrain_needs_a_local_model(tmp_db):
    from agent_eval.ml import selftrain

    with pytest.raises(ValueError, match="local model"):
        selftrain.self_train(selftrain.SelfTrainConfig(base_model="sim-base", experiment="x"))
