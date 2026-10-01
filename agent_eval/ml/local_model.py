"""Local Hugging Face / PyTorch code models, served through LangChain.

Model spec syntax (``--model``):

    hf:<repo-id-or-path>                  e.g. hf:Qwen/Qwen2.5-Coder-0.5B-Instruct
    hf:<repo-id-or-path>+<adapter-dir>    the same base plus a LoRA adapter from self-training

The model is loaded once per process with ``transformers`` (on CUDA, Apple
Silicon MPS, or CPU), wrapped as a LangChain ``ChatHuggingFace`` over a
``HuggingFacePipeline``, and called through the same ``invoke()`` interface the
loop uses for API models. Token counts come from the model's own tokenizer.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

from agent_eval.ml.device import chat_prompt_ids, pick_device, pick_dtype, seed_everything

PREFIX = "hf:"
MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "768"))


@dataclass(frozen=True)
class ModelSpec:
    base: str
    adapter: str | None = None

    def __str__(self) -> str:
        return PREFIX + self.base + (f"+{self.adapter}" if self.adapter else "")


def is_local(model: str) -> bool:
    return model.startswith(PREFIX)


def parse_spec(model: str) -> ModelSpec:
    if not is_local(model):
        raise ValueError(f"not a local model spec: {model!r} (expected {PREFIX}<repo-or-path>[+<adapter>])")
    body = model[len(PREFIX):]
    base, _, adapter = body.partition("+")
    if not base:
        raise ValueError(f"missing base model in {model!r}")
    return ModelSpec(base=base, adapter=adapter or None)


def with_adapter(model: str, adapter: str | None) -> str:
    """The spec for `model`'s base with a different (or no) adapter."""
    return str(ModelSpec(parse_spec(model).base, adapter))


class LocalChatModel:
    """One loaded model. Generation is serialized with a lock: a single model
    instance is not safe to call from several threads at once."""

    def __init__(self, spec: ModelSpec):
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.spec = spec
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(spec.base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.base, dtype=pick_dtype(self.device))
        if spec.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, spec.adapter)
            model = model.merge_and_unload()  # plain model again: faster generation
        model.to(self.device).eval()
        self.model = model
        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer,
                        device=self.device, return_full_text=False)
        self.chat = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=self.tokenizer)
        self._lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def invoke(self, system: str, user: str, temperature: float = 0.0, seed: int = 0,
               max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[str, int, int]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        gen = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen.update(do_sample=False)
        with self._lock:
            seed_everything(seed)
            out = self.chat.invoke(messages, pipeline_kwargs=gen)
        prompt_ids = chat_prompt_ids(
            self.tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        text = out.content if isinstance(out.content, str) else str(out.content)
        return text, len(prompt_ids), self.count_tokens(text)


@lru_cache(maxsize=2)
def load(model: str) -> LocalChatModel:
    return LocalChatModel(parse_spec(model))


def release() -> None:
    """Drop cached models (frees memory before fine-tuning on the same device)."""
    load.cache_clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
