"""Image-aware two-pass extraction for the VLM.

Per example: Pass 1 generate (with image inputs), Pass 2 a single teacher-forced forward over
prompt+generated tokens WITH the same vision inputs and output_hidden_states=True. Caches the
activation at each step-preceding and answer-preceding position across all backbone layers,
writing the SAME activations.dat / index.parquet / meta.json schema as the text pipeline (so the
shared analysis layer works unchanged) plus gen.jsonl for labeling/inspection.

Combined into one loop (not split S1/S2) because re-processing images for a separate pass is
wasteful; the go/no-go teacher-forcing match rate is reported the same way.

Batching: examples are processed `batch_size` at a time. Pass 1 uses LEFT padding (required so
batched generate() continues from the same column for every row); pass 2 re-pads the resulting
prompt+generation sequences with RIGHT padding (simpler position bookkeeping -- real content stays
at the same offsets from position 0 regardless of what's padded on after it). batch_size=1 exactly
reproduces the original unbatched code path (padding is a no-op on a batch of one).
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


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@torch.no_grad()
def build_dataset(
    model,
    processor,
    examples,
    device: str,
    out_dir: Path,
    model_name: str,
    step_mode: str = "think",
    max_new_tokens: int = 2048,
    checkpoint_every: int = 50,
    batch_size: int = 1,
    cached_gen_ids: Optional[Dict[str, List[int]]] = None,
) -> Dict:
    """Run the two-pass build over `examples`; write memmap + index + meta + gen.jsonl.

    cached_gen_ids: example_id -> gen_ids, e.g. reloaded from a previous run's gen.jsonl. When
    every example in a batch has a cached entry, pass 1 (generation, the expensive part) is
    skipped entirely and those ids are teacher-forced straight into pass 2 -- lets a run recover
    from a pass-2/write failure (e.g. a disk-quota hit during the final memmap flush) without
    re-paying for generation.

    Returns a summary dict (tf_match_rate, N positions, step stats, answer-detect rate).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    n_hidden, hidden_size = backbone_dims(model)
    tok.padding_side = "left"  # batched generate() needs every row's real prompt to end at the same column

    gen_f = open(out_dir / "gen.jsonl", "w")
    all_vecs: List[np.ndarray] = []
    index_rows: List[Dict] = []
    tf_match, tf_total = 0, 0
    step_counts, n_answer = [], 0
    row = 0
    n_done = 0

    for batch in _chunks(examples, batch_size):
        texts, images_batch = [], []
        for ex in batch:
            messages = build_messages(ex, step_mode=step_mode)
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images_batch.append(ex.images or None)
        images_arg = images_batch if any(im is not None for im in images_batch) else None
        inputs = processor(text=texts, images=images_arg, return_tensors="pt", padding=True).to(device)
        padded_prompt_len = inputs["input_ids"].shape[1]
        real_lens = inputs["attention_mask"].sum(dim=1).tolist()  # per-example true prompt length

        # Pass 1: batched greedy generation -- skipped entirely if every example in this batch
        # already has a cached gen_ids (recovery path).
        all_cached = cached_gen_ids is not None and all(ex.example_id in cached_gen_ids for ex in batch)
        gen = None if all_cached else model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)

        per_ex = []
        for bi, ex in enumerate(batch):
            real_len = int(real_lens[bi])
            real_prompt_ids = inputs["input_ids"][bi, padded_prompt_len - real_len:].tolist()
            if cached_gen_ids is not None and ex.example_id in cached_gen_ids:
                gen_ids = list(cached_gen_ids[ex.example_id])
            else:
                gen_ids = gen[bi, padded_prompt_len:].tolist()
                if eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id)]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)

            mk = parse_markers(ex.example_id, gen_ids, tok, real_len,
                               step_mode=step_mode, answer_mark=ANSWER_MARK)
            step_counts.append(mk.n_steps)
            n_answer += int(mk.answer is not None)

            real_mm = None
            if "mm_token_type_ids" in inputs:
                real_mm = inputs["mm_token_type_ids"][bi, padded_prompt_len - real_len:]
            per_ex.append(dict(ex=ex, gen_ids=gen_ids, real_prompt_ids=real_prompt_ids,
                               real_mm=real_mm, real_len=real_len, gen_text=gen_text, mk=mk))

        # Pass 2: batched teacher-forced forward with the SAME vision inputs.
        full_ids_list = [p["real_prompt_ids"] + p["gen_ids"] for p in per_ex]
        max_len2 = max(len(f) for f in full_ids_list)
        full_batch = torch.full((len(per_ex), max_len2), pad_id, dtype=torch.long, device=device)
        attn_batch = torch.zeros((len(per_ex), max_len2), dtype=torch.long, device=device)
        mm_batch = None
        if per_ex[0]["real_mm"] is not None:
            mm_batch = torch.zeros((len(per_ex), max_len2), dtype=per_ex[0]["real_mm"].dtype, device=device)
        for bi, p in enumerate(per_ex):
            L = len(full_ids_list[bi])
            full_batch[bi, :L] = torch.tensor(full_ids_list[bi], dtype=torch.long, device=device)
            attn_batch[bi, :L] = 1
            if mm_batch is not None:
                mm_batch[bi, :p["real_len"]] = p["real_mm"]  # generated-token region stays 0 (text)

        vision_kwargs = {k: v for k, v in inputs.items()
                         if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        if mm_batch is not None:
            vision_kwargs["mm_token_type_ids"] = mm_batch

        out = model(input_ids=full_batch, attention_mask=attn_batch, **vision_kwargs,
                    output_hidden_states=True, use_cache=False)
        logits = out.logits
        hs = out.hidden_states  # tuple len n_hidden, each [B, max_len2, hidden]

        for bi, p in enumerate(per_ex):
            ex, gen_ids, real_len, gen_text, mk = p["ex"], p["gen_ids"], p["real_len"], p["gen_text"], p["mk"]
            full_ids = full_ids_list[bi]
            seq_len = len(full_ids)

            for q in range(real_len, seq_len):
                tf_total += 1
                tf_match += int(int(torch.argmax(logits[bi, q - 1]).item()) == full_ids[q])

            for ptype, step_id, pos in _iter_positions(mk, seq_len):
                vec = torch.stack([hs[l][bi, pos] for l in range(n_hidden)])  # [n_hidden, hidden]
                all_vecs.append(vec.float().cpu().numpy().astype(np.float16))
                index_rows.append({"row": row, "example_id": ex.example_id, "position_type": ptype,
                                   "step_id": step_id, "token_index": pos, "n_steps": mk.n_steps,
                                   "correct": -1})
                row += 1

            gen_f.write(json.dumps(VLMGenResult(
                example_id=ex.example_id, dataset=ex.dataset, question_type=ex.question_type,
                gold_answer=ex.gold_answer, choices=ex.choices, meta=ex.meta, prompt_len=real_len,
                gen_ids=gen_ids, gen_text=gen_text, n_steps=mk.n_steps,
                has_answer=mk.answer is not None,
            ).to_json()) + "\n")

            n_done += 1
            if n_done % checkpoint_every == 0:
                gen_f.flush()

        del inputs, gen, out, hs, logits, full_batch, attn_batch
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
