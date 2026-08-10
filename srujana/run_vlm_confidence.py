#!/usr/bin/env python3
"""VLM Phase 2: fit + sanity-check Metric A (distance-based) and Metric B (trained classifier)
on the shared activation cache, sharing one train/calib/test split.

    PYTHONPATH=srujana python3 srujana/run_vlm_confidence.py --dataset mathvista --step-mode marker
"""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.confidence import (
    ece,
    brier,
    evaluate_confidence,
    fit_distance_model,
    fit_metric_b,
    fit_platt,
    score_examples,
    split_examples,
    subset,
)
from pipeline.features import load_examples
from pipeline.persist import save_json
from pipeline.predictor import build_features


def _report(name, y_test, p_raw_test, p_calib_test):
    from sklearn.metrics import roc_auc_score
    if len(set(y_test.tolist())) < 2:
        print(f"{name:>18}  insufficient test labels (n={len(y_test)})")
        return {}
    auc_raw = float(roc_auc_score(y_test, p_raw_test))
    auc_cal = float(roc_auc_score(y_test, p_calib_test))
    print(f"{name:>18}  AUC(raw)={auc_raw:.3f}  AUC(calib)={auc_cal:.3f}  "
          f"ECE={ece(p_calib_test, y_test):.3f}  Brier={brier(p_calib_test, y_test):.3f}  n={len(y_test)}")
    return {"auc_raw": auc_raw, "auc_calib": auc_cal,
            "ece": ece(p_calib_test, y_test), "brier": brier(p_calib_test, y_test), "n": len(y_test)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--dataset", default="mathvista", choices=["mathvista", "mmmu", "cvbench"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--step-mode", default="marker", choices=["think", "marker", "paragraph"])
    ap.add_argument("--layer", type=int, default=-1, help="hidden-state index; -1 = final layer")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.split or 'default'}_{args.step_mode}"
    rd = config.run_dir(args.model, dataset=tag)
    examples, meta = load_examples(rd)
    layer = meta["n_hidden"] - 1 if args.layer < 0 else args.layer
    print(f"[VLM-Phase2] model={args.model} dataset={args.dataset} layer={layer} "
          f"examples={len(examples)}")

    splits = split_examples(examples, seed=args.seed)
    train, calib, test = (subset(examples, splits[k]) for k in ("train", "calib", "test"))
    print(f"split: train={len(train)} calib={len(calib)} test={len(test)}")

    out = {"model": args.model, "dataset": args.dataset, "layer": layer,
          "n_train": len(train), "n_calib": len(calib), "n_test": len(test), "metrics": {}}

    # ---------------- Metric A: distance-based, training-free ----------------
    dm = fit_distance_model(train, layer=layer, pca_dim=args.pca_dim, seed=args.seed)
    print(f"\nMetric A: mu/sigma fit on {sum(1 for e in train.values() if e.correct == 1)} "
          f"train-correct examples, {len(dm.mu)} step ordinals with enough support")

    scores_calib = score_examples(dm, calib)
    scores_test = score_examples(dm, test)
    print("\n=== Metric A sanity (raw exp(-delta/sigma), before calibration) ===")
    for key in ("final", "last_step", "cumulative"):
        r = evaluate_confidence(scores_test, test, key=key)
        print(f"{key:>12}  AUC={r['auc']:.3f}  n={r['n']}")

    y_calib = [calib[eid].correct for eid in calib if scores_calib[eid]["final"] is not None]
    p_calib = [scores_calib[eid]["final"] for eid in calib if scores_calib[eid]["final"] is not None]
    y_test_a = [test[eid].correct for eid in test if scores_test[eid]["final"] is not None]
    p_test_a = [scores_test[eid]["final"] for eid in test if scores_test[eid]["final"] is not None]
    import numpy as np
    print("\n=== Metric A (calibrated, Platt-fit on calib) ===")
    if len(set(y_calib)) >= 2 and y_test_a:
        platt_a = fit_platt(np.array(p_calib), np.array(y_calib))
        p_calib_test_a = platt_a.apply(np.array(p_test_a))
        out["metrics"]["A_final"] = _report("A (final, calib)", np.array(y_test_a),
                                            np.array(p_test_a), p_calib_test_a)
    else:
        print("insufficient calib labels for Platt fit")

    # ---------------- Metric B: paper activation feature sets ----------------
    print("\n=== Metric B paper activation features ===")
    for kind in ("early_step", "late_trajectory", "final_state", "step_count"):
        Xtr, ytr = build_features(train, kind, layer)
        Xca, yca = build_features(calib, kind, layer)
        Xte, yte = build_features(test, kind, layer)
        if len(Xtr) < 10 or len(set(ytr.tolist())) < 2:
            print(f"{kind:>16}  insufficient train data (n={len(Xtr)})")
            continue
        fitted_b2, fit_info = fit_metric_b(Xtr, ytr, pca_dim=args.pca_dim, seed=args.seed)
        p_raw_calib = fitted_b2.predict_proba(Xca) if len(Xca) else None
        p_raw_test = fitted_b2.predict_proba(Xte)
        if p_raw_calib is not None and len(set(yca.tolist())) >= 2:
            platt_b2 = fit_platt(p_raw_calib, yca)
            p_calib_test = platt_b2.apply(p_raw_test)
            out["metrics"][f"B_{kind}"] = _report(f"B {kind}", yte, p_raw_test, p_calib_test)
        else:
            print(f"{kind:>16}  insufficient calib data for Platt fit")

    path = save_json(out, rd / "metrics_phase2_confidence.json")
    print(f"\n[VLM-Phase2] saved metrics -> {path}")


if __name__ == "__main__":
    main()
