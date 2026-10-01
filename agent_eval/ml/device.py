"""Pick the best available PyTorch device: CUDA, Apple Silicon (MPS), or CPU."""

from __future__ import annotations

import os


def pick_device() -> str:
    """AGENT_EVAL_DEVICE overrides; otherwise cuda > mps > cpu."""
    forced = os.getenv("AGENT_EVAL_DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """bf16 on GPUs that support it (half the memory of fp32), fp32 on CPU."""
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def chat_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Token ids of a chat prompt ending in the assistant turn. transformers 5
    returns a BatchEncoding from apply_chat_template(tokenize=True); older
    versions return a plain list -- normalize both."""
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if out and isinstance(out[0], list):  # batched form
        out = out[0]
    return list(out)


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
