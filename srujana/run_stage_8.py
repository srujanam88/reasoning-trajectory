#!/usr/bin/env python3
"""Stage 8: trajectory-based steering (paper §5.1, Figure 3a).

Fits the ideal trajectory (PCA + per-step mu/sigma) on ALL correct train examples, holds
out a slice of train purely for threshold tuning (percentile shortlist, confirmed via a
small number of real regeneration passes -- not a combinatorial grid, since corrections
compound and each candidate needs its own generation pass), then evaluates the chosen
thresholds once on the untouched test split, stratified by original step count.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np

from pipeline import config
from pipeline.features import load_examples
from pipeline.generate import GenResult
from pipeline.label import answers_match, extract_answer_after_hash
from pipeline.steer_trajectory import fit_ideal_trajectory, generate_with_trajectory_steering, tune_thresholds

MID_CENTER_HS = 16  # mid-depth layer default (paper's MID band is centered on layer 15)


def _load_gens(rd):
    gens = {}
    for line in open(rd / "gen.jsonl"):
        g = GenResult.from_json(json.loads(line))
        gens[g.example_id] = g
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--layer", type=int, default=MID_CENTER_HS)
    ap.add_argument("--tune-n", type=int, default=200, help="train examples held out for threshold tuning")
    ap.add_argument("--tune-eval-n", type=int, default=15, help="subset of the tune slice used for the real regeneration confirmation pass")
    ap.add_argument("--test-n", type=int, default=40, help="test subsample for final evaluation")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--lookahead-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.train_split}")
    test_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.test_split}")

    train_examples, train_meta = load_examples(train_rd)
    test_examples, _ = load_examples(test_rd)
    n_hidden = train_meta["n_hidden"]
    print(f"[S8] model={args.model}  layer={args.layer}  train_n={len(train_examples)}  test_n={len(test_examples)}")

    rng = random.Random(args.seed)
    train_ids = sorted(train_examples.keys())
    rng.shuffle(train_ids)
    tune_ids = set(train_ids[: args.tune_n])

    # mu/sigma/PCA fit on the FULL correct train pool (only *threshold tuning* is held out).
    traj = fit_ideal_trajectory(train_examples, args.layer, pca_dim=128)
    print(f"[S8] ideal trajectory fit; steps with sufficient support: {sorted(traj.mu.keys())}")

    tune_examples = {eid: train_examples[eid] for eid in tune_ids}
    candidates = tune_thresholds(traj, tune_examples, percentiles=(50, 60, 70, 80, 90))

    train_gens = _load_gens(train_rd)
    test_gens = _load_gens(test_rd)

    model, tok, device = config.load_model_and_tokenizer(args.model)
    # NOTE: unlike Stage 7's s^(l) = answer-step (where negative alpha "prolongs" by
    # subtracting the termination direction), low_rank_correction's (mu_j - z) already
    # points FROM the current activation TOWARD the ideal mean -- alpha must be positive
    # to move toward mu_j (correcting deviation); a negative alpha would push away from it.
    alpha = abs(args.alpha)

    def eval_candidate(cand, eids, gens, examples):
        baseline = {eid: examples[eid].correct for eid in eids}
        new_correct = {}
        for eid in eids:
            g = gens[eid]
            result = generate_with_trajectory_steering(
                model, tok, g.prompt_ids, traj,
                cand["delta_thresh"], cand["cum_thresh"], alpha,
                n_hidden, device, rank=args.rank, max_new_tokens=args.max_new_tokens,
                lookahead_tokens=args.lookahead_tokens,
            )
            pred = extract_answer_after_hash(result["gen_text"])
            new_correct[eid] = int(answers_match(pred, g.gold_answer))
        base_acc = 100.0 * sum(baseline.values()) / len(eids)
        new_acc = 100.0 * sum(new_correct.values()) / len(eids)
        n_orig_correct = sum(1 for eid in eids if baseline[eid] == 1)
        preserved = sum(1 for eid in eids if baseline[eid] == 1 and new_correct.get(eid) == 1)
        preservation = 100.0 * preserved / n_orig_correct if n_orig_correct else float("nan")
        return base_acc, new_acc, preservation

    tune_eval_candidates = [eid for eid in tune_ids if eid in train_gens and train_examples[eid].correct in (0, 1)]
    tune_eval_ids = sorted(rng.sample(tune_eval_candidates, min(args.tune_eval_n, len(tune_eval_candidates))))

    print(f"\n[S8] shortlisting thresholds on {len(tune_eval_ids)} tune examples...")
    best = None
    for cand in candidates:
        base_acc, new_acc, preservation = eval_candidate(cand, tune_eval_ids, train_gens, train_examples)
        gain = new_acc - base_acc
        ok = preservation >= 97.0 or np.isnan(preservation)
        print(f"  p={cand['percentile']:>3}  base={base_acc:.1f}%  new={new_acc:.1f}%  gain={gain:+.1f}pp  "
              f"preservation={preservation:.1f}%  {'OK' if ok else 'REJECT (preservation<97)'}")
        if ok and (best is None or gain > best[1]):
            best = (cand, gain)

    if best is None:
        print("[S8] no candidate met preservation>=97; falling back to the most permissive percentile")
        best = (candidates[-1], None)
    chosen = best[0]
    print(f"\n[S8] chosen threshold percentile: {chosen['percentile']}")

    test_candidates = [eid for eid in test_examples if eid in test_gens and test_examples[eid].correct in (0, 1)]
    test_ids = sorted(rng.sample(test_candidates, min(args.test_n, len(test_candidates))))
    buckets = defaultdict(list)
    for eid in test_ids:
        buckets[test_examples[eid].n_steps].append(eid)

    print("\n=== S8 SANITY: trajectory steering by step count ===")
    print(f"{'steps':>6} {'n':>4} {'before':>8} {'after':>8} {'delta':>9} {'preserve':>9}")
    for k in sorted(buckets):
        eids = buckets[k]
        base_acc, new_acc, preservation = eval_candidate(chosen, eids, test_gens, test_examples)
        print(f"{k:>6} {len(eids):>4} {base_acc:>7.2f}% {new_acc:>7.2f}% {new_acc-base_acc:>+8.2f}pp {preservation:>8.1f}%")

    print("\npaper targets: 6-step 75.44->83.04 (+7.60pp) | 7-step 67.69->75.38 (+7.69pp) | "
          "preservation>=97% | near-zero effect for <=5 steps")


if __name__ == "__main__":
    main()
