#!/usr/bin/env python3
"""Stage 6: correctness predictor (trajectory features) + baselines."""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import config
from pipeline.features import load_examples
from pipeline.predictor import build_features, fit_eval, layer_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--layer", type=int, default=-1, help="hidden-state index for fixed-layer sets; -1=final")
    ap.add_argument("--sweep", action="store_true", help="sweep all layers for late_trajectory peak")
    args = ap.parse_args()

    rd = config.run_dir(args.model)
    examples, meta = load_examples(rd)
    n_lbl = sum(1 for e in examples.values() if e.correct in (0, 1))
    n_pos = sum(1 for e in examples.values() if e.correct == 1)
    L = meta["n_hidden"] - 1 if args.layer < 0 else args.layer
    print(f"[S6] model={args.model}  labeled_examples={n_lbl}  (correct={n_pos})  layer={L}")

    print("\n=== S6 SANITY: correctness predictor AUC ===")
    rows = [
        ("step_count (baseline)", "step_count", None),
        ("early_step", "early_step", L),
        ("final_state (PCA128)", "final_state", L),
        ("late_trajectory (PCA128)", "late_trajectory", L),
    ]
    for name, kind, layer in rows:
        X, y = build_features(examples, kind, layer if layer is not None else 0)
        pca = None if kind == "step_count" else 128
        r = fit_eval(X, y, pca_dim=pca)
        auc = r["auc"]
        print(f"{name:>26} : AUC={auc:.3f}  (n={r['n']}, C={r.get('C')}, dim={r.get('dim', 1)})")

    print("\npaper targets (Instruct 8B): trajectory 0.852 | step-count 0.649 | "
          "final-state ~0.81 | early-step ~0.63")

    if args.sweep:
        layers = list(range(meta["n_hidden"]))
        per_layer, peak = layer_sweep(examples, "late_trajectory", layers)
        print("\n=== S6 late_trajectory layer sweep ===")
        if peak is not None:
            print(f"peak layer L{peak}: AUC={per_layer[peak]['auc']:.3f}  (paper peak ~L29)")
            aucs = [r["auc"] for r in per_layer.values() if not np.isnan(r["auc"])]
            print(f"mean over layers: {np.mean(aucs):.3f}  (paper ~0.83)")


if __name__ == "__main__":
    main()
