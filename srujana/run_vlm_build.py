#!/usr/bin/env python3
"""VLM Phase-0/1: generate + two-pass activation extraction for a multimodal dataset.

Writes the shared memmap schema (activations.dat / index.parquet / meta.json) + gen.jsonl.
Run on a GPU box. The teacher-forcing match rate is the go/no-go gate for VLM extraction.

Example (pilot):
    PYTHONPATH=srujana python3 srujana/run_vlm_build.py --dataset mathvista --n 20 --pilot
"""
from __future__ import annotations

import argparse
import json

from pipeline.vlm import config as vcfg
from pipeline.vlm.datasets import load_vlm_dataset
from pipeline.vlm.extract import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-thinking")
    ap.add_argument("--dataset", default="mathvista", choices=["mathvista", "mmmu", "cvbench"])
    ap.add_argument("--split", default=None, help="dataset split (default: testmini/validation/test)")
    ap.add_argument("--n", type=int, default=None, help="subsample size (None = full split)")
    ap.add_argument("--step-mode", default="think", choices=["think", "marker", "paragraph"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1, help="examples per generate()/forward() batch")
    ap.add_argument("--pilot", action="store_true", help="print sample generations for inspection")
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.split or 'default'}_{args.step_mode}"
    rd = vcfg.run_dir(args.model, dataset=tag)
    print(f"[VLM] loading dataset={args.dataset} split={args.split} n={args.n} ...", flush=True)
    examples = load_vlm_dataset(args.dataset, args.split, args.n, args.seed)
    print(f"[VLM] model={args.model} dataset={args.dataset} split={args.split} "
          f"n={len(examples)} step_mode={args.step_mode}")

    model, processor, device = vcfg.load_vlm(args.model)
    n_hidden, hidden = vcfg.backbone_dims(model)
    print(f"[VLM] device={device} n_hidden={n_hidden} hidden={hidden}")

    meta = build_dataset(model, processor, examples, device, rd, args.model,
                         step_mode=args.step_mode, max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size)

    size_mb = (rd / "activations.dat").stat().st_size / 1e6
    print("\n=== VLM BUILD SANITY ===")
    print(f"examples                 : {meta['n_examples']}")
    print(f"teacher-forcing match    : {meta['tf_match_rate']:.4f}  (GO/NO-GO; ~1.0 expected)")
    print(f"steps/example mean/median: {meta['mean_steps']:.2f} / {meta['median_steps']:.0f}")
    print(f"answer marker detected   : {meta['n_with_answer']}/{meta['n_examples']}")
    print(f"cached positions (N)     : {meta['N']}")
    print(f"activation shape         : [{meta['N']}, {meta['n_hidden']}, {meta['hidden_size']}] fp16")
    print(f"memmap size              : {size_mb:.1f} MB")
    print(f"saved                    : {rd}")

    if args.pilot:
        print("\n=== PILOT: sample generations (inspect step structure) ===")
        for line in list(open(rd / "gen.jsonl"))[:4]:
            g = json.loads(line)
            print("-" * 70)
            print(f"{g['example_id']} | gold={g['gold_answer']} | n_steps={g['n_steps']} "
                  f"| has_answer={g['has_answer']}")
            print(g["gen_text"][:800])


if __name__ == "__main__":
    main()
