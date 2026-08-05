"""Appendix-A fixed-form prompt and dataset loading.

The prompt is applied identically (raw text, no per-model chat template) to all models,
matching the paper's "hold prompting constant". ``final_answer_mark = "####"``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

FINAL_ANSWER_MARK = "####"

# Verbatim from the paper (Appendix A) / reference impl src/utils.py:1024 template "cot".
PROMPT_TEMPLATE = (
    'You are a helpful assistant that solves problems step by step with each step '
    'signified by "Step [step_number]: ".\n'
    "Always provide your final answer after #### at the end.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    'Please solve this step by step, putting each step after "Step [step_number]: " '
    "and always provide your final answer after ####.\n"
    "\n"
    "Solution:\n"
    "\n"
)


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


@dataclass
class Example:
    example_id: str
    question: str
    gold_answer: str  # normalized final answer (post-#### for GSM8K)
    raw_answer: str   # full gold solution text


def _gsm8k_gold(answer_field: str) -> str:
    """GSM8K gold solutions end in '#### <number>'."""
    if "####" in answer_field:
        return answer_field.split("####", 1)[1].strip()
    return answer_field.strip()


def load_gsm8k(split: str = "train", n: Optional[int] = None, seed: int = 42) -> List[Example]:
    """Load GSM8K via HF datasets; optionally take a seeded random subsample of size n."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idxs = list(range(len(ds)))
    if n is not None and n < len(ds):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(idxs, n))

    out: List[Example] = []
    for i in idxs:
        item = ds[i]
        out.append(
            Example(
                example_id=f"gsm8k_{split}_{i}",
                question=item["question"],
                gold_answer=_gsm8k_gold(item["answer"]),
                raw_answer=item["answer"],
            )
        )
    return out
