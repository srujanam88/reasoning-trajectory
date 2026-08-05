"""Stage 5: trajectory distance analysis.

Euclidean and cosine distances between consecutive step activations at the final layer,
stratified by correctness, with bootstrapped 95% CIs. Transitions (paper Fig 2a):
    Step1 -> Step2,  2nd-last -> Last,  Last -> Answer marker.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .features import ExampleActs

TRANSITIONS = ["step1_2", "secondlast_last", "last_answer"]


def _euclid(a, b):
    return float(np.linalg.norm(a - b))


def _cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


def example_transitions(e: ExampleActs, layer: int) -> Dict[str, Dict[str, float]]:
    steps = sorted(e.steps)
    out: Dict[str, Dict[str, float]] = {}
    if len(steps) >= 2:
        a, b = e.step_vec(steps[0], layer), e.step_vec(steps[1], layer)
        out["step1_2"] = {"euc": _euclid(a, b), "cos": _cosine_dist(a, b)}
        a, b = e.step_vec(steps[-2], layer), e.step_vec(steps[-1], layer)
        out["secondlast_last"] = {"euc": _euclid(a, b), "cos": _cosine_dist(a, b)}
    if steps and e.answer is not None:
        a, b = e.step_vec(steps[-1], layer), e.answer_vec(layer)
        out["last_answer"] = {"euc": _euclid(a, b), "cos": _cosine_dist(a, b)}
    return out


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 42):
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.mean(vals)), float(lo), float(hi))


def analyze(examples: Dict[str, ExampleActs], layer: int, metric: str = "euc",
            n_boot: int = 1000, seed: int = 42) -> Dict:
    """Per transition: correct/incorrect mean + 95% CI, delta(I-C), and CI-overlap flag."""
    buckets: Dict[str, Dict[int, List[float]]] = {t: {0: [], 1: []} for t in TRANSITIONS}
    for e in examples.values():
        if e.correct not in (0, 1):
            continue
        tr = example_transitions(e, layer)
        for t, d in tr.items():
            v = d[metric]
            if not np.isnan(v):
                buckets[t][e.correct].append(v)

    result = {}
    for t in TRANSITIONS:
        c = np.array(buckets[t][1])  # correct
        i = np.array(buckets[t][0])  # incorrect
        cm, clo, chi = _bootstrap_ci(c, n_boot, seed)
        im, ilo, ihi = _bootstrap_ci(i, n_boot, seed)
        overlap = not (chi < ilo or ihi < clo) if (len(c) and len(i)) else None
        result[t] = {
            "n_correct": len(c), "n_incorrect": len(i),
            "correct_mean": cm, "correct_ci": (clo, chi),
            "incorrect_mean": im, "incorrect_ci": (ilo, ihi),
            "delta_I_minus_C": (im - cm) if (len(c) and len(i)) else np.nan,
            "ci_overlap": overlap,
        }
    return result
