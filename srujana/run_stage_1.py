#!/usr/bin/env python3
"""Stage 1: prompt + generation + offset-based marker parsing.

Generates GSM8K CoT completions, parses Step/#### positions, prints sanity numbers, and
persists generations to outputs/<model>__gsm8k/gen.jsonl for Stages 2-3.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from pipeline import config
from pipeline.generate import generate_batch
from pipeline.markers import parse_markers, step_positions_by_token_id
from pipeline.prompting import load_gsm8k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    examples = load_gsm8k(split=args.split, n=args.n, seed=args.seed)
    print(f"[S1] model={args.model}  split={args.split}  n={len(examples)}")

    model, tok, device = config.load_model_and_tokenizer(args.model)
    print(f"[S1] device={device}  n_layers={model.config.num_hidden_layers}  d={model.config.hidden_size}")

    gens = generate_batch(model, tok, examples, device,
                          max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    # Parse markers + collect stats.
    step_counts, with_steps, with_answer, ans_agree_ex = [], 0, 0, 0
    offset_tok_agree_num, offset_tok_agree_den = 0, 0
    with open(rd / "gen.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g.to_json()) + "\n")
            mk = parse_markers(g.example_id, g.gen_ids, tok, g.prompt_len)
            step_counts.append(mk.n_steps)
            with_steps += int(mk.n_steps > 0)
            with_answer += int(mk.answer is not None)
            # Cross-check offset-based step positions vs token-id-8468 matching (Llama vocab).
            off = {s.abs_prec_pos for s in mk.steps}
            tid = set(step_positions_by_token_id(g.gen_ids, g.prompt_len))
            if off or tid:
                offset_tok_agree_num += len(off & tid)
                offset_tok_agree_den += len(off | tid)

    n = len(gens)
    print("\n=== S1 SANITY ===")
    print(f"examples                 : {n}")
    print(f"steps/example  mean/med  : {st.mean(step_counts):.2f} / {st.median(step_counts):.0f}")
    print(f"steps/example  min/max   : {min(step_counts)} / {max(step_counts)}")
    print(f"has >=1 Step marker      : {with_steps}/{n} ({100*with_steps/n:.0f}%)")
    print(f"has #### answer marker   : {with_answer}/{n} ({100*with_answer/n:.0f}%)")
    agree = (offset_tok_agree_num / offset_tok_agree_den) if offset_tok_agree_den else float("nan")
    print(f"offset vs token-id agree : {agree:.3f}  (Jaccard of step positions; ~1.0 expected on Llama)")
    print(f"saved                    : {rd / 'gen.jsonl'}")


if __name__ == "__main__":
    main()
