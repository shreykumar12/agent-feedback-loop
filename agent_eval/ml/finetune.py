"""LoRA fine-tuning of a causal code model on (chat prompt -> passing code) examples.

A plain PyTorch training loop over a PEFT LoRA adapter:
  * only the adapter's low-rank matrices train (~1% of parameters), which is
    what makes a 0.5B model trainable in 8 GB of unified memory
  * the loss covers only the answer tokens (prompt tokens are masked with -100),
    so the model learns to produce the fix, not to repeat the task
  * gradient accumulation (effective batch = batch_size * grad_accum), linear
    warmup + decay, gradient clipping, optional gradient checkpointing
  * a held-out slice of the examples reports eval loss before and after
The adapter is saved with ``save_pretrained`` and can be loaded back with the
model spec ``hf:<base>+<adapter-dir>``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_eval.ml.device import chat_prompt_ids, pick_device, seed_everything


@dataclass
class LoraTrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    lr: float = 2e-4
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    max_len: int = 1024
    warmup_fraction: float = 0.05
    eval_fraction: float = 0.1
    gradient_checkpointing: bool = True
    seed: int = 0
    max_steps: int | None = None  # cap optimizer steps (smoke tests)


def _encode(tokenizer, example: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    prompt_ids = chat_prompt_ids(tokenizer, example["messages"])
    target_ids = tokenizer(example["target"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None  # skip rather than truncate the answer
    return list(prompt_ids) + target_ids, [-100] * len(prompt_ids) + target_ids


def _collate(batch, pad_id: int):
    import torch

    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        attn[i, : len(x)] = 1
    return ids, labels, attn


def _mean_loss(model, data, cfg, pad_id, device) -> float | None:
    import torch

    if not data:
        return None
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(data), cfg.batch_size):
            ids, labels, attn = _collate(data[i:i + cfg.batch_size], pad_id)
            total += model(input_ids=ids.to(device), attention_mask=attn.to(device),
                           labels=labels.to(device)).loss.item() * len(data[i:i + cfg.batch_size])
    model.train()
    return total / len(data)


def finetune_lora(base: str, examples: list[dict], out_dir: str | Path, cfg: LoraTrainConfig | None = None,
                  log=print) -> dict:
    """Train a fresh LoRA adapter on `examples` and save it to `out_dir`."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or LoraTrainConfig()
    seed_everything(cfg.seed)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 weights everywhere but CUDA: MPS/CPU training is more stable in fp32,
    # and a 0.5B model is ~2 GB in fp32 -- fine on an 8 GB Mac.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM"))
    model.to(device).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    encoded = [e for e in (_encode(tokenizer, ex, cfg.max_len) for ex in examples) if e is not None]
    skipped = len(examples) - len(encoded)
    rng = random.Random(cfg.seed)
    rng.shuffle(encoded)
    n_eval = int(len(encoded) * cfg.eval_fraction) if len(encoded) >= 10 else 0
    eval_data, train_data = encoded[:n_eval], encoded[n_eval:]
    if not train_data:
        raise ValueError("no trainable examples (all longer than max_len?)")

    steps_per_epoch = math.ceil(len(train_data) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    warmup = max(1, int(total_steps * cfg.warmup_fraction))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    def lr_factor(s: int) -> float:  # linear warmup, then linear decay to 0
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    pad_id = tokenizer.pad_token_id
    eval_before = _mean_loss(model, eval_data, cfg, pad_id, device)
    log(f"  LoRA r={cfg.rank} on {device}: {trainable:,} trainable / {total_params:,} params; "
        f"{len(train_data)} train / {len(eval_data)} eval examples ({skipped} too long, skipped); "
        f"{total_steps} optimizer steps")

    start, step, losses, running = time.time(), 0, [], []
    done = False
    for epoch in range(cfg.epochs):
        rng.shuffle(train_data)
        for i in range(0, len(train_data), cfg.batch_size):
            ids, labels, attn = _collate(train_data[i:i + cfg.batch_size], pad_id)
            loss = model(input_ids=ids.to(device), attention_mask=attn.to(device), labels=labels.to(device)).loss
            (loss / cfg.grad_accum).backward()
            running.append(loss.item())
            if len(running) == cfg.grad_accum or i + cfg.batch_size >= len(train_data):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(sum(running) / len(running))
                running = []
                if step == 1 or step % 10 == 0 or step == total_steps:
                    log(f"    step {step}/{total_steps}  epoch {epoch + 1}  loss {losses[-1]:.4f}")
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    eval_after = _mean_loss(model, eval_data, cfg, pad_id, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    stats = {
        "base": base, "examples": len(examples), "trained_on": len(train_data), "eval_examples": len(eval_data),
        "skipped_too_long": skipped, "steps": step, "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None, "eval_loss_before": eval_before, "eval_loss_after": eval_after,
        "trainable_params": trainable, "total_params": total_params, "device": device,
        "seconds": round(time.time() - start, 1), "config": asdict(cfg),
    }
    (out / "training_stats.json").write_text(json.dumps(stats, indent=2))
    del model
    _free(device)
    return stats


def _free(device: str) -> None:
    import gc

    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
