"""Dataset loaders for the three multimodal benchmarks.

Each returns a list of VLMExample with a fully-formatted `question` (native query/prompt when
the dataset provides one), the gold answer, and the image(s). Correctness labeling lives in
label.py; prompt assembly (step instruction + answer marker) lives in prompting.py.
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VLMExample:
    example_id: str
    dataset: str            # "mathvista" | "mmmu" | "cvbench"
    images: List[Any]       # list of PIL images (>=1)
    question: str           # fully-formatted question text (native prompt where available)
    gold_answer: str
    question_type: str      # "multiple-choice" | "free_form" | "open"
    choices: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)


def _subsample(items, n, seed):
    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = [items[i] for i in sorted(rng.sample(range(len(items)), n))]
    return items


def load_mathvista(split: str = "testmini", n: Optional[int] = None, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset("AI4Math/MathVista", split=split)
    idxs = _subsample(list(range(len(ds))), n, seed)
    out = []
    for i in idxs:
        it = ds[i]
        qtype = it.get("question_type") or "free_form"
        out.append(VLMExample(
            example_id=f"mathvista_{it.get('pid', i)}",
            dataset="mathvista",
            images=[it["decoded_image"]] if it.get("decoded_image") is not None else [],
            question=it.get("query") or it["question"],  # native formatted prompt
            gold_answer=str(it.get("answer", "")),
            question_type="multiple-choice" if qtype == "multi_choice" else "free_form",
            choices=it.get("choices"),
            meta={"answer_type": it.get("answer_type"), "precision": it.get("precision"),
                  "unit": it.get("unit")},
        ))
    return out


def load_mmmu(split: str = "validation", n: Optional[int] = None, seed: int = 42,
              subjects: Optional[List[str]] = None):
    from datasets import get_dataset_config_names, load_dataset

    subjects = subjects or get_dataset_config_names("MMMU/MMMU")
    rows = []
    for subj in subjects:
        ds = load_dataset("MMMU/MMMU", subj, split=split)
        for it in ds:
            rows.append((subj, it))
    rows = _subsample(rows, n, seed)

    out = []
    for subj, it in rows:
        options = it.get("options")
        try:
            options = ast.literal_eval(options) if isinstance(options, str) else options
        except (ValueError, SyntaxError):
            options = None
        images = [it.get(f"image_{k}") for k in range(1, 8)]
        images = [im for im in images if im is not None]
        q = it["question"]
        if options:
            q = q + "\nOptions:\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))
        out.append(VLMExample(
            example_id=f"mmmu_{it.get('id', subj)}",
            dataset="mmmu",
            images=images,
            question=q,
            gold_answer=str(it.get("answer", "")),
            question_type=it.get("question_type") or "multiple-choice",
            choices=options,
            meta={"subject": subj, "subfield": it.get("subfield")},
        ))
    return out


def load_cvbench(split: str = "test", n: Optional[int] = None, seed: int = 42, config: str = "default"):
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/CV-Bench", config, split=split)
    idxs = _subsample(list(range(len(ds))), n, seed)
    out = []
    for i in idxs:
        it = ds[i]
        out.append(VLMExample(
            example_id=f"cvbench_{it.get('idx', i)}",
            dataset="cvbench",
            images=[it["image"]] if it.get("image") is not None else [],
            question=it.get("prompt") or it["question"],  # native prompt lists (A)..(D)
            gold_answer=str(it.get("answer", "")),         # e.g. "(C)"
            question_type="multiple-choice",
            choices=it.get("choices"),
            meta={"type": it.get("type"), "task": it.get("task")},
        ))
    return out


def load_vlm_dataset(name: str, split: Optional[str] = None, n: Optional[int] = None,
                     seed: int = 42):
    if name == "mathvista":
        return load_mathvista(split or "testmini", n, seed)
    if name == "mmmu":
        return load_mmmu(split or "validation", n, seed)
    if name == "cvbench":
        return load_cvbench(split or "test", n, seed)
    raise KeyError(f"unknown dataset '{name}' (mathvista|mmmu|cvbench)")
