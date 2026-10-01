"""Build a tiny, randomly initialized causal LM + tokenizer on disk.

It writes nonsense, but it exercises exactly the same code paths as a real
model (transformers loading, LangChain chat wrapper, LoRA fine-tuning, adapter
save/load) in seconds on a CPU, with no download. Tests and CI use it; on a
real machine you'd pass ``--model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval import config

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}</s>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>"]


def _corpus() -> list[str]:
    texts = []
    for path in config.SUITES.values():
        if Path(path).exists():
            for task in json.loads(Path(path).read_text()):
                texts += [task["prompt"], task.get("canonical_solution") or ""]
    return texts or ["def f(x):\n    return x\n"]


def build_tiny_model(out_dir: str | Path, vocab_size: int = 600, hidden: int = 64, layers: int = 2,
                     seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(_corpus(), trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                                     bos_token="<s>", eos_token="</s>")
    hf_tok.chat_template = CHAT_TEMPLATE
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=len(hf_tok), hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=4096, bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id, pad_token_id=hf_tok.pad_token_id, tie_word_embeddings=True,
    )
    LlamaForCausalLM(cfg).save_pretrained(out)
    hf_tok.save_pretrained(out)
    return out
