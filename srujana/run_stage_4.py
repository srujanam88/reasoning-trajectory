#!/usr/bin/env python3
"""Stage 4: step-identity linear probes (+ optional cross-model transfer)."""
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
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--transfer-from", default=None, help="source model name for cross-model transfer")
    ap.add_argument("--transfer-from-split", default=None,
                     help="split for --transfer-from (defaults to --split)")
    ap.add_argument("--min-per-class", type=int, default=10)
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    examples, meta = load_examples(rd)
    X, y = build_step_identity_matrix(examples)
    layers = list(range(meta["n_hidden"]))
    print(f"[S4] model={args.model}  positions={len(y)}  n_hidden={meta['n_hidden']}")

    res = run_step_probes(X, y, layers=layers, min_per_class=args.min_per_class)
    summ = best_layer_summary(res["acc"])

    print("\n=== S4 SANITY: step-identity probes (within-model) ===")
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
        src_split = args.transfer_from_split or args.split
        rd_src = config.run_dir(args.transfer_from, dataset=f"gsm8k_{src_split}")
        ex_src, _ = load_examples(rd_src)
        Xs, ys = build_step_identity_matrix(ex_src)
        tr = cross_model_transfer(Xs, ys, X, y, layers=layers, min_per_class=args.min_per_class)
        print(f"\n=== S4 cross-model transfer: {args.transfer_from} -> {args.model} ===")
        for label, per_layer in tr.items():
            if per_layer:
                bl = max(per_layer, key=per_layer.get)
                print(f"{str(label):>8}  best_acc={per_layer[bl]:.3f} @L{bl}  "
                      f"mean={np.mean(list(per_layer.values())):.3f}")


if __name__ == "__main__":
    main()
