#!/usr/bin/env python3
"""(1) Step-count distribution summary per split, (2) regenerate Fig 1b with a subset of
step classes. CPU-only; reads cached Stage-2 outputs, no GPU / no generation.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.features import load_examples
from pipeline.probes import build_step_identity_matrix, run_step_probes
from pipeline.viz import plot_probe_accuracy_by_layer

BUCKETS = [(2, 4), (5, 7), (8, 11), (12, 10**9)]
SUBSET_CLASSES = [1, 2, 3, 5, 8, "answer"]


def step_count_summary(model, split):
    rd = config.run_dir(model, dataset=f"gsm8k_{split}")
    df = pd.read_parquet(rd / "index.parquet")
    per_ex = df.drop_duplicates("example_id")[["example_id", "n_steps"]]
    n = len(per_ex)
    print(f"\n=== step-count distribution: {split} ({n} questions) ===")
    print(f"mean={per_ex.n_steps.mean():.2f}  median={per_ex.n_steps.median():.0f}  "
          f"min={per_ex.n_steps.min()}  max={per_ex.n_steps.max()}")
    print("bucket        count   %")
    for lo, hi in BUCKETS:
        c = int(((per_ex.n_steps >= lo) & (per_ex.n_steps <= hi)).sum())
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        print(f"  {label:<10} {c:>5}  {100*c/n:5.1f}%")
    print("per-step counts:", dict(sorted(per_ex.n_steps.value_counts().to_dict().items())))
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--plot-split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for split in args.splits:
        step_count_summary(args.model, split)

    # Regenerate Fig 1b (subset of classes) for the plot split.
    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.plot_split}")
    examples, meta = load_examples(rd)
    layers = list(range(meta["n_hidden"]))
    print(f"\n[plot] running step-identity probes over {meta['n_hidden']} layers "
          f"(subset {SUBSET_CLASSES})...")
    X, y = build_step_identity_matrix(examples)
    res = run_step_probes(X, y, layers=layers, seed=args.seed)
    out = rd / "plots" / "probe_accuracy_by_layer_subset.png"
    plot_probe_accuracy_by_layer(res["acc"], str(out), include_classes=SUBSET_CLASSES)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
