#!/usr/bin/env python3
"""Stage 9: reasoning-length control (paper §5.2, Figure 3b).

Reuses Stage 7's steering vectors/layers/hook infra (pipeline/steering.py), but the hook
stays armed for the WHOLE decode (every generated token gets alpha*s^(l) added), sweeping
|alpha| for SHORTEN (alpha>0) and PROLONG (alpha<0) at LAST/MID. Expect roughly graded
length change at low |alpha| with small accuracy cost, and qualitative breakdown
(repetition / hitting max_new_tokens without a natural EOS) at high |alpha|.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from pipeline import config
from pipeline.features import load_examples
from pipeline.generate import GenResult
from pipeline.label import answers_match, extract_answer_after_hash
from pipeline.steering import LAST_HS, MID_HS, compute_steering_vectors, generate_with_continuous_intervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.train_split}")
    test_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.test_split}")
    train_examples, train_meta = load_examples(train_rd)
    test_examples, _ = load_examples(test_rd)
    n_hidden = train_meta["n_hidden"]

    all_layers = sorted(set(LAST_HS) | set(MID_HS))
    vectors = compute_steering_vectors(train_examples, all_layers)
    print(f"[S9] model={args.model}  steering vectors computed for layers {all_layers}")

    gens = {}
    for line in open(test_rd / "gen.jsonl"):
        g = GenResult.from_json(json.loads(line))
        gens[g.example_id] = g

    candidates = [eid for eid, e in test_examples.items() if eid in gens and e.correct in (0, 1)]
    rng = random.Random(args.seed)
    subsample = sorted(rng.sample(candidates, min(args.n, len(candidates))))

    baseline_len = {eid: len(gens[eid].gen_ids) for eid in subsample}
    baseline_correct = {eid: test_examples[eid].correct for eid in subsample}
    base_acc = 100.0 * sum(baseline_correct.values()) / len(subsample)
    base_mean_len = float(np.mean(list(baseline_len.values())))

    model, tok, device = config.load_model_and_tokenizer(args.model)

    print(f"[S9] baseline: acc={base_acc:.2f}%  mean_len={base_mean_len:.1f} tokens  n={len(subsample)}")
    print("\n=== S9 SANITY: length-control sweep ===")
    print(f"{'config':>6} {'mode':>8} {'alpha':>6} {'mean_len':>9} {'d_len':>8} {'acc':>7} {'d_acc':>7} {'breakdown':>10}")

    for name, hs_indices in [("Last", LAST_HS), ("Mid", MID_HS)]:
        for mode, sign in [("PROLONG", -1), ("SHORTEN", 1)]:
            for a in args.alphas:
                alpha = sign * a
                lens, corrects, breakdowns = [], [], []
                for eid in subsample:
                    g = gens[eid]
                    result = generate_with_continuous_intervention(
                        model, tok, g.prompt_ids, hs_indices, vectors, alpha, n_hidden, device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    lens.append(len(result["gen_ids"]))
                    breakdowns.append(result["hit_budget"])
                    pred = extract_answer_after_hash(result["gen_text"])
                    corrects.append(int(answers_match(pred, g.gold_answer)))
                mean_len = float(np.mean(lens))
                acc = 100.0 * float(np.mean(corrects))
                brk = 100.0 * float(np.mean(breakdowns))
                print(f"{name:>6} {mode:>8} {a:>6.2f} {mean_len:>9.1f} {mean_len-base_mean_len:>+8.1f} "
                      f"{acc:>6.2f}% {acc-base_acc:>+6.2f}pp {brk:>9.1f}%")

    print("\npaper pattern: low |alpha| -> graded length change, ~1% or less accuracy cost; "
          "high |alpha| (>0.8) -> qualitative breakdown (repetition/non-termination)")


if __name__ == "__main__":
    main()
