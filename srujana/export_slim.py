#!/usr/bin/env python3
"""Export a slim (few-layer) activation file so distance / final-layer analysis can run
without the full ~1.5 GB memmap. RUN THIS WHERE activations.dat EXISTS (e.g. RunPod), then
sync the resulting slim_acts.npz (~45 MB/layer) back.

Example:
    PYTHONPATH=srujana python3 srujana/export_slim.py --model llama-3.1-8b-instruct --splits train test
    # default keeps only the final layer; pass --layers for more (e.g. --layers 29 32)
"""
from __future__ import annotations

import argparse

from pipeline import config
from pipeline.extract import export_slim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden-state indices to keep (default: final layer only)")
    args = ap.parse_args()

    for split in args.splits:
        rd = config.run_dir(args.model, dataset=f"gsm8k_{split}")
        out = export_slim(rd, layers=args.layers)
        mb = out.stat().st_size / 1e6
        print(f"[export] {split}: {out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
