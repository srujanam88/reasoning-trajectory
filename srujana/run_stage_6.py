#!/usr/bin/env python3
"""Stage 6: correctness predictor (trajectory features) + baselines."""
from __future__ import annotations

import argparse

import numpy as np

from pipeline import config
from pipeline.baselines import answer_marker_token_id, build_logit_lens_features, logit_lens_layer_sweep
from pipeline.features import load_examples
from pipeline.predictor import build_features, fit_eval, layer_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer", type=int, default=-1, help="hidden-state index for fixed-layer sets; -1=final")
    ap.add_argument("--sweep", action="store_true", help="sweep all layers for late_trajectory + logit-lens peak")
    ap.add_argument("--device", default="cpu",
                     help="device for the logit-lens model load (norm+lm_head unembed only; "
                          "CPU by default -- S3-S6 stay off the GPU clock per PLAN.md)")
    args = ap.parse_args()

    rd = config.run_dir(args.model, dataset=f"gsm8k_{args.split}")
    examples, meta = load_examples(rd)
    n_lbl = sum(1 for e in examples.values() if e.correct in (0, 1))
    n_pos = sum(1 for e in examples.values() if e.correct == 1)
    L = meta["n_hidden"] - 1 if args.layer < 0 else args.layer
    print(f"[S6] model={args.model}  split={args.split}  labeled_examples={n_lbl}  (correct={n_pos})  layer={L}")

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

    # Logit-lens baseline (paper Table 3): entropy / answer-marker-token rank / top-1
    # probability at the (last-step, answer) boundaries -- reuses Stage-2's cached
    # hidden states via an offline unembed, no new generation.
    ll_model, ll_tok, _ = config.load_model_and_tokenizer(args.model, device=args.device)
    marker_id = answer_marker_token_id(ll_tok)
    X, y = build_logit_lens_features(examples, L, ll_model, marker_id)
    r = fit_eval(X, y, pca_dim=None)
    print(f"{'logit_lens (entropy/rank/top1)':>26} : AUC={r['auc']:.3f}  "
          f"(n={r['n']}, C={r.get('C')}, dim={r.get('dim', 1)})")

    print("\npaper targets (Instruct 8B): trajectory 0.852 | step-count 0.649 | "
          "logit-lens 0.765 | final-state ~0.81 | early-step ~0.63")

    if args.sweep:
        layers = list(range(meta["n_hidden"]))
        per_layer, peak = layer_sweep(examples, "late_trajectory", layers)
        print("\n=== S6 late_trajectory layer sweep ===")
        if peak is not None:
            print(f"peak layer L{peak}: AUC={per_layer[peak]['auc']:.3f}  (paper peak ~L29)")
            aucs = [r["auc"] for r in per_layer.values() if not np.isnan(r["auc"])]
            print(f"mean over layers: {np.mean(aucs):.3f}  (paper ~0.83)")

        ll_per_layer, ll_peak = logit_lens_layer_sweep(examples, layers, ll_model, marker_id, fit_eval)
        print("\n=== S6 logit_lens layer sweep ===")
        if ll_peak is not None:
            print(f"peak layer L{ll_peak}: AUC={ll_per_layer[ll_peak]['auc']:.3f}  "
                  f"(paper best-config 0.765)")


if __name__ == "__main__":
    main()
