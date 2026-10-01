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


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
