#!/usr/bin/env python3
"""Re-plot Fig 1b for any subset of step classes from the SAVED probe_accuracy.json --
no activations.dat and no recompute needed. Requires that run_visualize.py has been run
once (on the box with activations) to produce plots/probe_accuracy.json.
"""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.viz import load_probe_accuracy, plot_probe_accuracy_by_layer


def _parse_class(tok: str):
    return tok if tok.lower() in ("answer", "ans") else int(tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--classes", nargs="+", default=["1", "2", "3", "5", "8", "answer"],
                    help="step ids and/or 'answer' to include")
    ap.add_argument("--out", default="probe_accuracy_by_layer_subset.png")
    args = ap.parse_args()

    plots_dir = config.run_dir(args.model, dataset=f"gsm8k_{args.split}") / "plots"
    acc = load_probe_accuracy(str(plots_dir / "probe_accuracy.json"))
    include = [_parse_class(c) for c in args.classes]
    out = plot_probe_accuracy_by_layer(acc, str(plots_dir / args.out), include_classes=include)
    print(f"[replot] classes={include}  ->  {out}")


if __name__ == "__main__":
    main()
