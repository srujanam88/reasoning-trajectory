"""Batched greedy Pass-1 generation (Stage 1).

Deterministic greedy decoding (do_sample=False), KV cache on. Returns, per example, the
unpadded prompt token ids and the generated token ids (trailing pad/eos stripped) plus the
decoded text. Left padding (set on the tokenizer) makes batching correct.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import torch

from .prompting import Example, build_prompt


@dataclass
class GenResult:
    example_id: str
    question: str
    gold_answer: str
    prompt_len: int
    prompt_ids: List[int]
    gen_ids: List[int]
    gen_text: str

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "GenResult":
        return GenResult(**d)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    examples: List[Example],
    device: str,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> List[GenResult]:
    results: List[GenResult] = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [build_prompt(ex.question) for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        input_len = input_ids.shape[1]

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        for row, ex in enumerate(batch):
            prompt_len = int(attn[row].sum().item())
            # Left padding => real prompt tokens are the last `prompt_len` of the input row.
            prompt_ids = input_ids[row, input_len - prompt_len : input_len].tolist()
            gen = out[row, input_len:].tolist()
            # Strip trailing pad; cut at first eos (exclusive).
            gen = [t for t in gen if t != pad_id]
            if eos_id in gen:
                gen = gen[: gen.index(eos_id)]
            gen_text = tokenizer.decode(gen, skip_special_tokens=True)
            results.append(
                GenResult(
                    example_id=ex.example_id,
                    question=ex.question,
                    gold_answer=ex.gold_answer,
                    prompt_len=prompt_len,
                    prompt_ids=prompt_ids,
                    gen_ids=gen,
                    gen_text=gen_text,
                )
            )
    return results
