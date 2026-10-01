"""A learned verifier, built from scratch in PyTorch: does this code pass the hidden tests?

The model reads the task prompt and a candidate solution and predicts the
probability that the sandbox would accept it -- *without running anything*.
In the loop it reranks best-of-N candidates so the most promising one is the
one that gets tested (and counts as the attempt).

Everything is hand-built, no pretrained weights:
  * ByteTokenizer   -- raw UTF-8 bytes + 3 special tokens (no vocabulary to learn)
  * VerifierNet     -- byte embeddings grouped into patches of `patch` bytes (4x
                       shorter sequences, ~16x cheaper attention), learned position
                       embeddings, pre-norm transformer
                       blocks with a hand-written multi-head self-attention, then
                       [CLS] + masked-mean pooling into a single logit
  * train_verifier  -- AdamW, cosine schedule, class-balanced BCE, early stopping
                       on validation AUC, gradient clipping
  * evaluate        -- accuracy, ROC-AUC, Brier score, calibration error (ECE), and
                       best-of-N selection accuracy vs. a random-pick baseline

Labels come from the deterministic sandbox, so they are ground truth: reference
solutions (pass), validated mutants (fail), and every stored attempt.
Splits are by *task*, so the reported numbers measure generalization to
problems the verifier never saw, not memorized programs.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from agent_eval.ml.device import pick_device, seed_everything

# --- data -------------------------------------------------------------------


@dataclass
class Example:
    task_id: str
    prompt: str
    code: str
    label: int  # 1 = passes all hidden tests
    source: str = ""  # "canonical", "mutant", "attempt"


def examples_from_tasks(tasks) -> list[Example]:
    """Reference solutions are positives; mutants are negatives (the suites'
    100% mutation score guarantees every mutant fails at least one test)."""
    out = []
    for t in tasks:
        if t.canonical_solution:
            out.append(Example(t.task_id, t.prompt, t.canonical_solution, 1, "canonical"))
        out += [Example(t.task_id, t.prompt, m, 0, "mutant") for m in t.mutants]
    return out


def examples_from_db(task_ids: set[str] | None = None) -> list[Example]:
    """Every stored attempt, labeled by its sandbox verdict."""
    from agent_eval import storage

    out = []
    with storage.get_connection() as conn:
        rows = conn.execute(
            "SELECT a.task_id, t.prompt, a.code, a.passed FROM attempts a JOIN tasks t ON t.task_id = a.task_id"
        ).fetchall()
    conn.close()
    for r in rows:
        if task_ids is None or r["task_id"] in task_ids:
            if r["code"].strip():
                out.append(Example(r["task_id"], r["prompt"], r["code"], int(r["passed"]), "attempt"))
    return out


def dedupe(examples: list[Example]) -> list[Example]:
    seen, out = {}, []
    for ex in examples:
        key = (ex.task_id, hashlib.sha1(ex.code.strip().encode()).hexdigest())
        if key in seen:
            # Conflicting labels for identical code would be a sandbox bug; keep the first.
            continue
        seen[key] = ex.label
        out.append(ex)
    return out


def split_by_task(examples: list[Example], val_fraction: float = 0.15, seed: int = 0):
    """Deterministic task-level split so no task appears in both sides."""
    tasks = sorted({e.task_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(tasks)
    n_val = max(1, int(len(tasks) * val_fraction)) if len(tasks) > 1 else 0
    val_tasks = set(tasks[:n_val])
    return [e for e in examples if e.task_id not in val_tasks], [e for e in examples if e.task_id in val_tasks]


# --- tokenizer ----------------------------------------------------------------

class ByteTokenizer:
    PAD, CLS, SEP = 0, 1, 2
    OFFSET = 3
    vocab_size = 256 + OFFSET

    def __init__(self, max_len: int = 1024, prompt_budget: int = 256):
        self.max_len = max_len
        self.prompt_budget = prompt_budget

    def _bytes(self, text: str, limit: int) -> list[int]:
        return [b + self.OFFSET for b in text.encode("utf-8", "replace")[:limit]]

    def encode(self, prompt: str, code: str) -> list[int]:
        p = self._bytes(prompt, self.prompt_budget)
        c = self._bytes(code, self.max_len - len(p) - 2)
        return [self.CLS, *p, self.SEP, *c]

    def batch(self, pairs: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode(p, c) for p, c in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids, ids != self.PAD


# --- model --------------------------------------------------------------------

@dataclass
class VerifierConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    dropout: float = 0.1
    max_len: int = 1024
    prompt_budget: int = 256
    patch: int = 4  # bytes per token after the patch embedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)        # (b, h, t, t)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))  # ignore padding keys
        attn = self.dropout(scores.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, cfg: VerifierConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                 nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.mlp(self.ln2(x))


class VerifierNet(nn.Module):
    def __init__(self, cfg: VerifierConfig):
        super().__init__()
        self.cfg = cfg
        self.byte = nn.Embedding(ByteTokenizer.vocab_size, cfg.d_model, padding_idx=ByteTokenizer.PAD)
        self.patch = nn.Linear(cfg.patch * cfg.d_model, cfg.d_model)  # concat p byte embeddings -> 1 token
        self.pos = nn.Embedding(cfg.max_len // cfg.patch + 1, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch
        pad = (-ids.shape[1]) % p
        if pad:
            ids = F.pad(ids, (0, pad), value=ByteTokenizer.PAD)
            mask = F.pad(mask, (0, pad), value=False)
        b, t = ids.shape
        x = self.patch(self.byte(ids).view(b, t // p, p * self.cfg.d_model))
        mask = mask.view(b, t // p, p).any(-1)  # a patch is real if any of its bytes is
        positions = torch.arange(t // p, device=ids.device)
        x = x + self.pos(positions)[None]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        m = mask.unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([x[:, 0], mean], dim=-1)).squeeze(-1)  # logits


# --- metrics ------------------------------------------------------------------

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


def expected_calibration_error(labels: list[int], probs: list[float], bins: int = 10) -> float:
    total, ece = len(labels), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            ece += len(idx) / total * abs(conf - acc)
    return ece


def selection_metrics(examples: list[Example], probs: list[float]) -> dict:
    """Best-of-N view: for each task with at least one passing and one failing
    candidate, does the top-scored candidate pass? Compared with picking at random."""
    by_task: dict[str, list[tuple[float, int]]] = {}
    for ex, p in zip(examples, probs, strict=True):
        by_task.setdefault(ex.task_id, []).append((p, ex.label))
    hits = rand = n = 0
    for cands in by_task.values():
        labels = [y for _, y in cands]
        if 0 < sum(labels) < len(labels):
            n += 1
            top = max(p for p, _ in cands)
            tied = [y for p, y in cands if p == top]
            hits += sum(tied) / len(tied)  # ties broken uniformly at random, never by label
            rand += sum(labels) / len(labels)
    return {"tasks": n, "top1_pass_rate": hits / n if n else None, "random_pass_rate": rand / n if n else None}


# --- training -----------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 3
    seed: int = 0
    log_every: int = 0  # batches; 0 = per epoch only


@dataclass
class TrainResult:
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    val: dict = field(default_factory=dict)


class Verifier:
    """A trained model + tokenizer; scores (prompt, code) pairs."""

    def __init__(self, net: VerifierNet, device: str | None = None):
        self.device = device or pick_device()
        self.net = net.to(self.device).eval()
        self.tok = ByteTokenizer(net.cfg.max_len, net.cfg.prompt_budget)

    @torch.no_grad()
    def score_batch(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        out = []
        for i in range(0, len(pairs), batch_size):
            ids, mask = self.tok.batch(pairs[i:i + batch_size])
            out += torch.sigmoid(self.net(ids.to(self.device), mask.to(self.device))).tolist()
        return out

    def score(self, prompt: str, code: str) -> float:
        return self.score_batch([(prompt, code)])[0]

    def save(self, path: str | Path, metrics: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.net.cfg), "state_dict": self.net.state_dict(),
                    "metrics": metrics or {}}, path)
        return path


def evaluate(verifier: Verifier, examples: list[Example]) -> dict:
    if not examples:
        return {"n": 0}
    probs = verifier.score_batch([(e.prompt, e.code) for e in examples])
    labels = [e.label for e in examples]
    preds = [int(p >= 0.5) for p in probs]
    return {
        "n": len(examples),
        "positives": sum(labels),
        "accuracy": sum(int(p == y) for p, y in zip(preds, labels, strict=True)) / len(labels),
        "auc": roc_auc(labels, probs),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(labels),
        "ece": expected_calibration_error(labels, probs),
        **{f"select_{k}": v for k, v in selection_metrics(examples, probs).items()},
    }


def train_verifier(train: list[Example], val: list[Example], model_cfg: VerifierConfig | None = None,
                   cfg: TrainConfig | None = None, log=print) -> tuple[Verifier, TrainResult]:
    model_cfg, cfg = model_cfg or VerifierConfig(), cfg or TrainConfig()
    if not train:
        raise ValueError("no training examples")
    seed_everything(cfg.seed)
    device = pick_device()
    if device == "cpu":
        import os

        torch.set_num_threads(max(1, os.cpu_count() or 1))
    net = VerifierNet(model_cfg).to(device)
    tok = ByteTokenizer(model_cfg.max_len, model_cfg.prompt_budget)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(train) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    n_pos = sum(e.label for e in train)
    pos_weight = torch.tensor((len(train) - n_pos) / max(1, n_pos), device=device)  # balance classes
    rng = random.Random(cfg.seed)

    best_auc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for b, i in enumerate(range(0, len(order), cfg.batch_size)):
            batch = [train[j] for j in order[i:i + cfg.batch_size]]
            ids, mask = tok.batch([(e.prompt, e.code) for e in batch])
            y = torch.tensor([float(e.label) for e in batch], device=device)
            loss = F.binary_cross_entropy_with_logits(net(ids.to(device), mask.to(device)), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
            if cfg.log_every and b % cfg.log_every == 0:
                log(f"    epoch {epoch} batch {b} loss {loss.item():.4f}")
        verifier = Verifier(net, device)
        val_metrics = evaluate(verifier, val) if val else {}
        net.train()
        row = {"epoch": epoch, "train_loss": total / len(train), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        auc = val_metrics.get("auc")
        log(f"  epoch {epoch:>2}  train loss {row['train_loss']:.4f}"
            + (f"  val auc {auc:.3f}  val acc {val_metrics['accuracy']:.3f}" if auc is not None else ""))
        score = auc if auc is not None else -row["train_loss"]
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  early stop (no val improvement for {cfg.patience} epochs)")
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    verifier = Verifier(net, device)
    return verifier, TrainResult(best_epoch=best_epoch, history=history,
                                 val=evaluate(verifier, val) if val else {})


@lru_cache(maxsize=4)
def load_verifier_cached(path: str) -> Verifier:
    return load_verifier(path)


def load_verifier(path: str | Path, device: str | None = None) -> Verifier:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    net = VerifierNet(VerifierConfig(**ckpt["config"]))
    net.load_state_dict(ckpt["state_dict"])
    return Verifier(net, device)


def checkpoint_metrics(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True).get("metrics", {})


def summarize(metrics: dict) -> str:
    def f(v, pct=False):
        if v is None:
            return "-"
        return f"{v:.1%}" if pct else f"{v:.3f}"
    return (f"n={metrics.get('n', 0)} pos={metrics.get('positives', 0)}  acc {f(metrics.get('accuracy'), True)}  "
            f"AUC {f(metrics.get('auc'))}  Brier {f(metrics.get('brier'))}  ECE {f(metrics.get('ece'))}  |  "
            f"best-of-N on {metrics.get('select_tasks', 0)} tasks: verifier pick passes "
            f"{f(metrics.get('select_top1_pass_rate'), True)} vs random {f(metrics.get('select_random_pass_rate'), True)}")


__all__ = ["Example", "Verifier", "VerifierConfig", "TrainConfig", "train_verifier", "evaluate",
           "load_verifier", "examples_from_tasks", "examples_from_db", "dedupe", "split_by_task"]
