#!/usr/bin/env python3
"""Stage 3: correctness labeling.

Parses the final answer from each Stage-1 generation, compares to GSM8K gold, and writes
the per-position correctness label back into the Stage-2 index.parquet (column `correct`).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from pipeline import config
from pipeline.generate import GenResult
from pipeline.label import answers_match, extract_answer_after_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    args = ap.parse_args()

    rd = config.run_dir(args.model)
    gens = [GenResult.from_json(json.loads(line)) for line in open(rd / "gen.jsonl")]

    correct_by_ex, parsed_ok = {}, 0
    for g in gens:
        pred = extract_answer_after_hash(g.gen_text)
        parsed_ok += int(pred is not None)
        correct_by_ex[g.example_id] = bool(answers_match(pred, g.gold_answer))

    n = len(gens)
    n_correct = sum(correct_by_ex.values())

    # Write labels into the Stage-2 index if present.
    idx_path = rd / "index.parquet"
    labeled_positions = None
    if idx_path.exists():
        df = pd.read_parquet(idx_path)
        df["correct"] = df["example_id"].map(lambda e: int(correct_by_ex.get(e, -1)))
        df.to_parquet(idx_path)
        labeled_positions = int((df["correct"] >= 0).sum())

    print("\n=== S3 SANITY ===")
    print(f"examples                 : {n}")
    print(f"answer parsed            : {parsed_ok}/{n} ({100*parsed_ok/n:.0f}%)")
    print(f"GSM8K accuracy           : {n_correct}/{n} ({100*n_correct/n:.1f}%)")
    print(f"class balance (C/I)      : {n_correct} / {n - n_correct}")
    if labeled_positions is not None:
        print(f"index positions labeled  : {labeled_positions}")
    else:
        print("index.parquet not found  : run stage 2 to attach labels to positions")


if __name__ == "__main__":
    main()
