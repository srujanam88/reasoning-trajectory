#!/usr/bin/env python3
"""Stage 2: two-pass hidden-state extraction to a memmap.

Loads Stage-1 generations, runs a single teacher-forced forward per example, and caches
activations at each Step-preceding and answer-preceding position across all hidden layers.
"""
from __future__ import annotations

import argparse
import json
import os

from pipeline import config
from pipeline.extract import extract
from pipeline.generate import GenResult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    gen_path = rd / "gen.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"[S2] missing {gen_path}; run stage 1 first.")

    gens = [GenResult.from_json(json.loads(line)) for line in open(gen_path)]
    print(f"[S2] model={args.model}  examples={len(gens)}")

    model, tok, device = config.load_model_and_tokenizer(args.model)
    meta = extract(model, tok, gens, device, rd, args.model)

    size_mb = (rd / "activations.dat").stat().st_size / 1e6
    print("\n=== S2 SANITY ===")
    print(f"cached positions (N)     : {meta['N']}")
    print(f"activation shape         : [{meta['N']}, {meta['n_hidden']}, {meta['hidden_size']}]  fp16")
    print(f"memmap size              : {size_mb:.1f} MB  ({size_mb/max(len(gens),1):.2f} MB/example)")
    tf = meta["tf_match_rate"]
    print(f"teacher-forcing match    : {tf:.4f}  (argmax logits[q-1]==gen[q]; ~1.0 expected)")
    print(f"saved                    : {rd / 'activations.dat'}, index.parquet, meta.json")


if __name__ == "__main__":
    main()
