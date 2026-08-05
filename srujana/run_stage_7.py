#!/usr/bin/env python3
"""Stage 7: predictor-gated activation steering (paper §4.3, Table 4 Prolong rows).

Steering vectors s^(l) are computed from the TRAIN split (pooled over every
(example, step_k) pair, all layers). A gating predictor (late_trajectory features,
Section 4.2's architecture) is fit on TRAIN and applied to TEST's own already-cached
activations to decide, per example, whether an "impending failure" is predicted --
entirely offline, no new generation needed for the gating decision itself.

Only examples selected for steering are regenerated (Always: 100% of the subsample;
Gated: only the ~12% flagged). The one-shot intervention reuses the ORIGINAL (unsteered)
generation's answer-marker position directly from Stage 2's index.parquet (token_index of
the 'answer' row) -- since decoding up to that point is deterministic/greedy and
unperturbed, no live marker re-detection is needed.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from pipeline import config
from pipeline.extract import load_activations
from pipeline.features import load_examples
from pipeline.generate import GenResult
from pipeline.label import answers_match, extract_answer_after_hash
from pipeline.predictor import build_features, fit_deployable
from pipeline.steering import LAST_HS, MID_HS, compute_steering_vectors, generate_with_one_shot_intervention


def _late_trajectory_row(e, layer):
    steps = sorted(e.steps)
    if not steps or e.answer is None:
        return None
    return np.concatenate([e.step_vec(steps[-1], layer), e.answer_vec(layer)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--n", type=int, default=30, help="test subsample size")
    ap.add_argument("--alpha", type=float, default=0.05, help="|alpha|; Prolong uses -alpha")
    ap.add_argument("--gating-layer", type=int, default=29, help="hidden-states index for the gating predictor (paper peak layer)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.train_split}")
    test_rd = config.run_dir(args.model, dataset=f"gsm8k_{args.test_split}")

    train_examples, train_meta = load_examples(train_rd)
    test_examples, test_meta = load_examples(test_rd)
    n_hidden = train_meta["n_hidden"]
    print(f"[S7] model={args.model}  train_n={len(train_examples)}  test_n={len(test_examples)}  n_hidden={n_hidden}")

    # 1. Steering vectors from train (LAST + MID layers, pooled over all step->answer pairs).
    all_layers = sorted(set(LAST_HS) | set(MID_HS))
    vectors = compute_steering_vectors(train_examples, all_layers)
    print(f"[S7] steering vectors computed for layers {all_layers}")

    # 2. Gating predictor: fit on train's late_trajectory features, deployable on test.
    Xg, yg = build_features(train_examples, "late_trajectory", args.gating_layer)
    gating_model, ginfo = fit_deployable(Xg, yg, pca_dim=128, seed=args.seed)
    print(f"[S7] gating predictor: cv_auc={ginfo['cv_auc']:.3f}  C={ginfo['C']}  n={ginfo['n']}")

    # 3. Test-split generation inputs (prompt_ids/gen_ids) + answer-marker position.
    gens = {}
    for line in open(test_rd / "gen.jsonl"):
        g = GenResult.from_json(json.loads(line))
        gens[g.example_id] = g
    _, index, _ = load_activations(test_rd)
    answer_pos = {
        row.example_id: int(row.token_index)
        for row in index.itertuples(index=False)
        if row.position_type == "answer"
    }

    # 4. Candidate subsample: needs a known answer position, last-step + answer
    #    activations (for gating), a known correctness label, and its original generation.
    candidates = [
        eid for eid, e in test_examples.items()
        if eid in answer_pos and eid in gens and e.correct in (0, 1)
        and e.steps and e.answer is not None
    ]
    rng = random.Random(args.seed)
    subsample = sorted(rng.sample(candidates, min(args.n, len(candidates))))
    print(f"[S7] test candidates={len(candidates)}  subsample={len(subsample)}")

    baseline_correct = {eid: test_examples[eid].correct for eid in subsample}
    baseline_acc = 100.0 * sum(baseline_correct.values()) / len(subsample)

    # 5. Offline gating decision (predict_proba < 0.5 => predicted incorrect => flag).
    Xtest_g = np.stack([_late_trajectory_row(test_examples[eid], args.gating_layer) for eid in subsample])
    p_correct = gating_model.predict_proba(Xtest_g)
    flagged = {eid: bool(p_correct[i] < 0.5) for i, eid in enumerate(subsample)}
    n_flagged = sum(flagged.values())
    print(f"[S7] flagged (predicted-incorrect) = {n_flagged}/{len(subsample)} "
          f"({100*n_flagged/len(subsample):.1f}%, paper ~12.3%)")

    model, tok, device = config.load_model_and_tokenizer(args.model)
    alpha = -abs(args.alpha)  # Prolong: alpha < 0

    print("\n=== S7 SANITY: Table 4 (Prolong rows) ===")
    print(f"{'config':>14} {'mode':>8} {'acc':>8} {'delta_vs_base':>14} {'n_steered':>10}")
    print(f"{'baseline':>14} {'-':>8} {baseline_acc:>7.2f}% {'--':>14} {0:>10}")

    for name, hs_indices in [("Last", LAST_HS), ("Mid", MID_HS)]:
        for mode in ["always", "gated"]:
            new_correct = dict(baseline_correct)
            n_steered = 0
            for eid in subsample:
                fire = True if mode == "always" else flagged[eid]
                if not fire:
                    continue
                n_steered += 1
                g = gens[eid]
                pos = answer_pos[eid]
                rel = pos - g.prompt_len
                result = generate_with_one_shot_intervention(
                    model, tok, g.prompt_ids, g.gen_ids, rel,
                    hs_indices, vectors, alpha, n_hidden, device,
                    max_new_tokens=args.max_new_tokens,
                )
                pred = extract_answer_after_hash(result["gen_text"])
                new_correct[eid] = int(answers_match(pred, g.gold_answer))
            acc = 100.0 * sum(new_correct.values()) / len(subsample)
            delta = acc - baseline_acc
            print(f"{name:>14} {mode:>8} {acc:>7.2f}% {delta:>+13.2f}pp {n_steered:>10}")

    print("\npaper targets (Always / Gated, pp delta): "
          "Prolong(Last) +0.45 / +0.76 | Prolong(Mid) +0.38 / +0.76")


if __name__ == "__main__":
    main()
