#!/usr/bin/env python3
"""VLM Phase 1b: trajectory distance analysis (correctness-stratified, bootstrapped CIs).

Same analysis as run_stage_5.py (pipeline/distances.py is dataset-agnostic) pointed at a VLM run
directory. Requires index.parquet's `correct` column to be populated -- run run_vlm_label.py first.

    PYTHONPATH=srujana python3 srujana/run_vlm_distances.py --dataset mathvista --step-mode marker
"""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.distances import analyze
from pipeline.features import load_examples
from pipeline.persist import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--dataset", default="mathvista", choices=["mathvista", "mmmu", "cvbench"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--step-mode", default="marker", choices=["think", "marker", "paragraph"])
    ap.add_argument("--layer", type=int, default=-1, help="hidden-state index; -1 = final layer")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.split or 'default'}_{args.step_mode}"
    rd = config.run_dir(args.model, dataset=tag)
    examples, meta = load_examples(rd)
    layer = meta["n_hidden"] - 1 if args.layer < 0 else args.layer
    print(f"[VLM-S5] model={args.model} dataset={args.dataset} step_mode={args.step_mode} "
          f"layer={layer} examples={len(examples)}")

    n_labeled = sum(1 for e in examples.values() if e.correct in (0, 1))
    if n_labeled == 0:
        raise SystemExit("No examples have a correct/incorrect label (index.parquet 'correct' "
                         "column is all -1). Run run_vlm_label.py on this run first.")

    out = {"model": args.model, "dataset": args.dataset, "step_mode": args.step_mode,
           "layer": layer, "n_boot": args.n_boot, "n_examples": len(examples), "metrics": {}}
    for metric, label in [("euc", "Euclidean"), ("cos", "Cosine")]:
        res = analyze(examples, layer=layer, metric=metric, n_boot=args.n_boot)
        out["metrics"][metric] = res
        print(f"\n=== VLM S5 SANITY: {label} distances (final layer) ===")
        print(f"{'transition':>18} {'correct':>10} {'incorrect':>10} {'d(I-C)':>9} {'CIoverlap':>10}")
        for t, r in res.items():
            ov = "-" if r["ci_overlap"] is None else ("yes" if r["ci_overlap"] else "NO")
            print(f"{t:>18} {r['correct_mean']:>10.2f} {r['incorrect_mean']:>10.2f} "
                  f"{r['delta_I_minus_C']:>9.2f} {ov:>10}  "
                  f"(nC={r['n_correct']}, nI={r['n_incorrect']})")
        print("note: paper Euclidean last->answer d(I-C) ~ -13.39 (Instruct 8B); "
              "'NO' overlap = significant divergence")

    path = save_json(out, rd / "metrics_s5_distances.json")
    print(f"\n[VLM-S5] saved metrics -> {path}")


if __name__ == "__main__":
    main()
