#!/usr/bin/env python3
"""Visualizations: t-SNE (Fig 1a/4), probe accuracy by layer (Fig 1b), predictor AUC by
layer (Fig 2b). CPU-only -- reads the cached Stage-2 activations, no GPU needed, so this
can run alongside any GPU-bound stage.
"""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.features import load_examples
from pipeline.predictor import layer_sweep
from pipeline.probes import build_step_identity_matrix, run_step_probes
from pipeline.viz import DEFAULT_TSNE_LAYERS, plot_auc_by_layer, plot_probe_accuracy_by_layer, plot_tsne_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tsne-layers", type=int, nargs="+", default=None)
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-max-points", type=int, default=3000)
    ap.add_argument("--auc-layer-kind", default="late_trajectory")
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--skip-probe-curve", action="store_true")
    ap.add_argument("--skip-auc-curve", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    plots_dir = rd / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    examples, meta = load_examples(rd)
    n_hidden = meta["n_hidden"]
    layers = list(range(n_hidden))
    print(f"[VIZ] model={args.model}  split={args.split}  n_examples={len(examples)}  n_hidden={n_hidden}")

    if not args.skip_tsne:
        tsne_layers = args.tsne_layers or [l for l in DEFAULT_TSNE_LAYERS if l < n_hidden] + [n_hidden - 1]
        tsne_layers = sorted(set(tsne_layers))
        print(f"[VIZ] t-SNE across layers {tsne_layers} (Fig 1a/4)...")
        out = plot_tsne_grid(examples, tsne_layers, str(plots_dir / "tsne_by_layer.png"),
                              perplexity=args.tsne_perplexity, seed=args.seed,
                              max_points=args.tsne_max_points)
        print(f"[VIZ] saved {out}")

    if not args.skip_probe_curve:
        print(f"[VIZ] step-identity probe accuracy across all {n_hidden} layers (Fig 1b)...")
        X, y = build_step_identity_matrix(examples)
        res = run_step_probes(X, y, layers=layers, seed=args.seed)
        out = plot_probe_accuracy_by_layer(res["acc"], str(plots_dir / "probe_accuracy_by_layer.png"))
        print(f"[VIZ] saved {out}")

    if not args.skip_auc_curve:
        print(f"[VIZ] correctness-predictor AUC across all {n_hidden} layers (Fig 2b, "
              f"kind={args.auc_layer_kind})...")
        per_layer, peak = layer_sweep(examples, args.auc_layer_kind, layers, seed=args.seed)
        if peak is not None:
            print(f"[VIZ] peak layer L{peak}: AUC={per_layer[peak]['auc']:.3f} (paper peak ~L29, 0.87)")
        out = plot_auc_by_layer(per_layer, str(plots_dir / "predictor_auc_by_layer.png"))
        print(f"[VIZ] saved {out}")

    print(f"\n[VIZ] all plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
