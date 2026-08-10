#!/usr/bin/env python3
"""VLM labeling: parse each generation's final answer, compare to gold, write correctness
into the shared index.parquet (column `correct`). CPU-only; reads gen.jsonl.

    PYTHONPATH=srujana python3 srujana/run_vlm_label.py --dataset mathvista --split testmini --step-mode marker
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from pipeline.vlm import config as vcfg
from pipeline.vlm.label import label_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-thinking")
    ap.add_argument("--dataset", default="mathvista", choices=["mathvista", "mmmu", "cvbench"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--step-mode", default="marker", choices=["marker", "paragraph"])
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.split or 'default'}_{args.step_mode}"
    rd = vcfg.run_dir(args.model, dataset=tag)
    gens = [json.loads(line) for line in open(rd / "gen.jsonl")]

    correct_by_ex, parsed_ok = {}, 0
    for g in gens:
        ok = label_example(g["dataset"], g["gen_text"], g["gold_answer"], g.get("choices"),
                           g["question_type"], g.get("meta"))
        parsed_ok += int(g["has_answer"])
        correct_by_ex[g["example_id"]] = bool(ok)

    n = len(gens)
    n_correct = sum(correct_by_ex.values())

    idx_path = rd / "index.parquet"
    labeled = None
    if idx_path.exists():
        df = pd.read_parquet(idx_path)
        df["correct"] = df["example_id"].map(lambda e: int(correct_by_ex.get(e, -1)))
        df.to_parquet(idx_path)
        labeled = int((df["correct"] >= 0).sum())

    print("\n=== VLM LABEL SANITY ===")
    print(f"dataset                  : {args.dataset} ({args.split})")
    print(f"examples                 : {n}")
    print(f"answer marker present    : {parsed_ok}/{n} ({100*parsed_ok/n:.0f}%)")
    print(f"accuracy                 : {n_correct}/{n} ({100*n_correct/n:.1f}%)")
    print(f"class balance (C/I)      : {n_correct} / {n - n_correct}")
    if labeled is not None:
        print(f"index positions labeled  : {labeled}")


if __name__ == "__main__":
    main()
