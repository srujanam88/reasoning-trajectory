"""Image-aware two-pass extraction for the VLM.

Per example: Pass 1 generate (with image inputs), Pass 2 a single teacher-forced forward over
prompt+generated tokens WITH the same vision inputs and output_hidden_states=True. Caches the
activation at each step-preceding and answer-preceding position across all backbone layers,
writing the SAME activations.dat / index.parquet / meta.json schema as the text pipeline (so the
shared analysis layer works unchanged) plus gen.jsonl for labeling/inspection.

Combined into one loop (not split S1/S2) because re-processing images for a separate pass is
wasteful; the go/no-go teacher-forcing match rate is reported the same way.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import backbone_dims
from .markers import parse_markers
from .prompting import ANSWER_MARK, build_messages


@dataclass
class VLMGenResult:
    example_id: str
    dataset: str
    question_type: str
    gold_answer: str
    choices: Optional[List[str]]
    meta: dict
    prompt_len: int
    gen_ids: List[int]
    gen_text: str
    n_steps: int
    has_answer: bool

    def to_json(self) -> dict:
        return asdict(self)


def _iter_positions(mk, seq_len):
    for s in mk.steps:
        if 0 <= s.abs_prec_pos < seq_len:
            yield ("step", s.step_id, s.abs_prec_pos)
    if mk.answer is not None and 0 <= mk.answer.abs_prec_pos < seq_len:
        yield ("answer", 0, mk.answer.abs_prec_pos)


@torch.no_grad()
def build_dataset(
    model,
    processor,
    examples,
    device: str,
    out_dir: Path,
    model_name: str,
    step_mode: str = "marker",
    max_new_tokens: int = 2048,
    checkpoint_every: int = 50,
) -> Dict:
    """Run the two-pass build over `examples`; write memmap + index + meta + gen.jsonl.

    Returns a summary dict (tf_match_rate, N positions, step stats, answer-detect rate).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    n_hidden, hidden_size = backbone_dims(model)

    gen_f = open(out_dir / "gen.jsonl", "w")
    all_vecs: List[np.ndarray] = []
    index_rows: List[Dict] = []
    tf_match, tf_total = 0, 0
    step_counts, n_answer = [], 0
    row = 0

    for ei, ex in enumerate(examples):
        messages = build_messages(ex, step_mode=step_mode)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=ex.images or None, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Pass 1: greedy generation.
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or eos_id)
        gen_ids = gen[0, input_len:].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        mk = parse_markers(ex.example_id, gen_ids, tok, input_len,
                           step_mode=step_mode, answer_mark=ANSWER_MARK)
        step_counts.append(mk.n_steps)
        n_answer += int(mk.answer is not None)

        # Pass 2: teacher-forced forward with the SAME vision inputs.
        full_ids = inputs["input_ids"][0].tolist() + gen_ids
        full = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(full)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        out = model(input_ids=full, attention_mask=attn, **vision_kwargs,
                    output_hidden_states=True, use_cache=False)
        logits = out.logits[0]
        hs = out.hidden_states  # tuple len n_hidden, each [1, seq, hidden]
        seq_len = len(full_ids)

        for q in range(input_len, seq_len):
            tf_total += 1
            tf_match += int(int(torch.argmax(logits[q - 1]).item()) == full_ids[q])

        for ptype, step_id, p in _iter_positions(mk, seq_len):
            vec = torch.stack([hs[l][0, p] for l in range(n_hidden)])  # [n_hidden, hidden]
            all_vecs.append(vec.float().cpu().numpy().astype(np.float16))
            index_rows.append({"row": row, "example_id": ex.example_id, "position_type": ptype,
                               "step_id": step_id, "token_index": p, "n_steps": mk.n_steps,
                               "correct": -1})
            row += 1

        gen_f.write(json.dumps(VLMGenResult(
            example_id=ex.example_id, dataset=ex.dataset, question_type=ex.question_type,
            gold_answer=ex.gold_answer, choices=ex.choices, meta=ex.meta, prompt_len=input_len,
            gen_ids=gen_ids, gen_text=gen_text, n_steps=mk.n_steps,
            has_answer=mk.answer is not None,
        ).to_json()) + "\n")

        if (ei + 1) % checkpoint_every == 0:
            gen_f.flush()
        del inputs, gen, out, hs, logits, full
        if device == "cuda":
            torch.cuda.empty_cache()

    gen_f.close()

    N = len(all_vecs)
    acts = np.memmap(out_dir / "activations.dat", dtype=np.float16, mode="w+",
                     shape=(N, n_hidden, hidden_size))
    for i, v in enumerate(all_vecs):
        acts[i] = v
    acts.flush()

    import pandas as pd
    pd.DataFrame(index_rows).to_parquet(out_dir / "index.parquet")

    meta = {"N": N, "n_hidden": n_hidden, "hidden_size": hidden_size, "model": model_name,
            "dtype": "float16", "step_mode": step_mode,
            "tf_match_rate": (tf_match / tf_total) if tf_total else None,
            "n_examples": len(examples), "n_with_answer": n_answer,
            "mean_steps": float(np.mean(step_counts)) if step_counts else 0.0,
            "median_steps": float(np.median(step_counts)) if step_counts else 0.0}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
