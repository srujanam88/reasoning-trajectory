#!/usr/bin/env python3
"""VLM Phase 1a: step-identity linear probes (+ optional cross-model transfer).

Same analysis as run_stage_4.py (pipeline/probes.py is dataset-agnostic -- it just reads the
memmap + index.parquet schema) pointed at a VLM run directory instead of a text-pipeline one.

    PYTHONPATH=srujana python3 srujana/run_vlm_probes.py --dataset mathvista --step-mode marker
"""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import config
from pipeline.features import load_examples
from pipeline.probes import (
    best_layer_summary,
    build_step_identity_matrix,
    cross_model_transfer,
    run_step_probes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--dataset", default="mathvista", choices=["mathvista", "mmmu", "cvbench"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--step-mode", default="marker", choices=["think", "marker", "paragraph"])
    ap.add_argument("--transfer-from", default=None, help="source model name for cross-model transfer")
    ap.add_argument("--transfer-from-dataset", default=None, help="dataset for --transfer-from (defaults to --dataset)")
    ap.add_argument("--transfer-from-step-mode", default=None, help="step-mode for --transfer-from (defaults to --step-mode)")
    ap.add_argument("--min-per-class", type=int, default=10)
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.split or 'default'}_{args.step_mode}"
    rd = config.run_dir(args.model, dataset=tag)
    examples, meta = load_examples(rd)
    X, y = build_step_identity_matrix(examples)
    layers = list(range(meta["n_hidden"]))
    print(f"[VLM-S4] model={args.model} dataset={args.dataset} step_mode={args.step_mode} "
          f"positions={len(y)} n_hidden={meta['n_hidden']}")

    res = run_step_probes(X, y, layers=layers, min_per_class=args.min_per_class)
    summ = best_layer_summary(res["acc"])

    print("\n=== VLM S4 SANITY: step-identity probes (within-model) ===")
    print(f"{'class':>8} {'best_layer':>10} {'best_acc':>9} {'mean_acc':>9}")
    for label in res["classes"]:
        bl, ba, ma = summ[label]
        print(f"{str(label):>8} {bl:>10} {ba:>9.3f} {ma:>9.3f}")

    # Shuffled-label control (~0.59 in the paper).
    sh = run_step_probes(X, y, layers=layers, min_per_class=args.min_per_class, shuffle_labels=True)
    sh_summ = best_layer_summary(sh["acc"])
    if sh_summ:
        mean_shuf = float(np.mean([m for (_, _, m) in sh_summ.values()]))
        print(f"\nshuffled-label control (avg mean acc): {mean_shuf:.3f}  (paper ~0.59)")

    if args.transfer_from:
        src_tag = (f"{args.transfer_from_dataset or args.dataset}_{args.split or 'default'}_"
                   f"{args.transfer_from_step_mode or args.step_mode}")
        rd_src = config.run_dir(args.transfer_from, dataset=src_tag)
        ex_src, _ = load_examples(rd_src)
        Xs, ys = build_step_identity_matrix(ex_src)
        tr = cross_model_transfer(Xs, ys, X, y, layers=layers, min_per_class=args.min_per_class)
        print(f"\n=== VLM S4 cross-model transfer: {args.transfer_from} -> {args.model} ===")
        for label, per_layer in tr.items():
            if per_layer:
                bl = max(per_layer, key=per_layer.get)
                print(f"{str(label):>8}  best_acc={per_layer[bl]:.3f} @L{bl}  "
                      f"mean={np.mean(list(per_layer.values())):.3f}")


if __name__ == "__main__":
    main()
