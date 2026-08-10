#!/usr/bin/env python3
"""Stage 5: trajectory distance analysis (correctness-stratified, bootstrapped CIs)."""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.distances import analyze
from pipeline.features import load_examples
from pipeline.persist import save_json


def _fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer", type=int, default=-1, help="hidden-state index; -1 = final layer")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    examples, meta = load_examples(rd)
    layer = meta["n_hidden"] - 1 if args.layer < 0 else args.layer
    print(f"[S5] model={args.model}  layer={layer}  examples={len(examples)}")

    out = {"model": args.model, "split": args.split, "layer": layer,
           "n_boot": args.n_boot, "n_examples": len(examples), "metrics": {}}
    for metric, label in [("euc", "Euclidean"), ("cos", "Cosine")]:
        res = analyze(examples, layer=layer, metric=metric, n_boot=args.n_boot)
        out["metrics"][metric] = res
        print(f"\n=== S5 SANITY: {label} distances (final layer) ===")
        print(f"{'transition':>18} {'correct':>10} {'incorrect':>10} {'d(I-C)':>9} {'CIoverlap':>10}")
        for t, r in res.items():
            ov = "-" if r["ci_overlap"] is None else ("yes" if r["ci_overlap"] else "NO")
            print(f"{t:>18} {r['correct_mean']:>10.2f} {r['incorrect_mean']:>10.2f} "
                  f"{r['delta_I_minus_C']:>9.2f} {ov:>10}  "
                  f"(nC={r['n_correct']}, nI={r['n_incorrect']})")
        print("note: paper Euclidean last->answer d(I-C) ~ -13.39 (Instruct 8B); "
              "'NO' overlap = significant divergence")

    path = save_json(out, rd / "metrics_s5_distances.json")
    print(f"\n[S5] saved metrics -> {path}")


if __name__ == "__main__":
    main()
