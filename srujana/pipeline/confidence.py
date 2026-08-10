"""Phase 2: two answer-confidence metrics on the shared VLM activation cache.

Metric A -- distance-based, training-free. mu_j/sigma_j per reasoning-step ordinal j are
estimated from TRAIN-CORRECT activations only (PCA-128 space). A new example's confidence at
step j is exp(-delta_j / sigma_j) where delta_j = ||z_j - mu_j|| -- how close it stays to the
"ideal" (correct) trajectory, sigma-normalized so it's comparable across steps/layers. No
fitting beyond mu/sigma.

Metric B -- trained classifier. pipeline/predictor.py's single-layer logistic regression, reused
verbatim, on predictor.py's early_step / late_trajectory / final_state activation feature sets.

Both metrics are fit on the SAME train/calib/test split (train_test_split below): mu/sigma and
the classifier are fit on train only; Platt scaling is fit on calib; everything is reported on
test, so Phase 3 comparisons are apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as _sk_split
from sklearn.preprocessing import StandardScaler

from .features import ExampleActs
from .predictor import FittedPredictor, fit_deployable

MIN_STEP_SUPPORT = 5  # need at least this many train-correct examples at a step ordinal to trust mu/sigma there


# --------------------------------- split ---------------------------------

def split_examples(
    examples: Dict[str, ExampleActs], train: float = 0.6, calib: float = 0.2, seed: int = 42
) -> Dict[str, List[str]]:
    """Stratified (by correctness) train/calib/test split over labeled example ids.

    Examples without a correct/incorrect label can't be used by either metric and are dropped.
    """
    ids = [k for k, e in examples.items() if e.correct in (0, 1)]
    labels = [examples[k].correct for k in ids]
    ids_train, ids_rest, y_train, y_rest = _sk_split(
        ids, labels, train_size=train, stratify=labels, random_state=seed
    )
    calib_frac_of_rest = calib / (1 - train)
    ids_calib, ids_test, _, _ = _sk_split(
        ids_rest, y_rest, train_size=calib_frac_of_rest, stratify=y_rest, random_state=seed
    )
    return {"train": ids_train, "calib": ids_calib, "test": ids_test}


def subset(examples: Dict[str, ExampleActs], ids: List[str]) -> Dict[str, ExampleActs]:
    return {i: examples[i] for i in ids}


# ----------------------------- Metric A: distance-based -----------------------------

@dataclass
class DistanceModel:
    scaler: StandardScaler
    pca: PCA
    mu: Dict[int, np.ndarray]      # step ordinal -> [pca_dim] mean (train-correct only)
    sigma: Dict[int, float]        # step ordinal -> scalar spread of ||z - mu|| (train-correct only)
    mu_answer: Optional[np.ndarray]
    sigma_answer: Optional[float]
    layer: int

    def _z(self, vec: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(vec[None, :]))[0]


def fit_distance_model(
    examples: Dict[str, ExampleActs], layer: int, pca_dim: int = 128, seed: int = 42
) -> DistanceModel:
    """mu_j, sigma_j per step from TRAIN-CORRECT activations only (per VLM_PLAN.md Phase 2)."""
    correct = {k: e for k, e in examples.items() if e.correct == 1}

    rows = []
    for e in correct.values():
        for sid in e.steps:
            rows.append(e.step_vec(sid, layer))
        if e.answer is not None:
            rows.append(e.answer_vec(layer))
    X = np.stack(rows).astype(np.float32)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n_comp = min(pca_dim, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xs)

    def _z(vec):
        return pca.transform(scaler.transform(vec[None, :]))[0]

    per_step: Dict[int, List[np.ndarray]] = {}
    answer_zs = []
    for e in correct.values():
        for sid in e.steps:
            per_step.setdefault(sid, []).append(_z(e.step_vec(sid, layer)))
        if e.answer is not None:
            answer_zs.append(_z(e.answer_vec(layer)))

    mu, sigma = {}, {}
    for sid, zs in per_step.items():
        if len(zs) < MIN_STEP_SUPPORT:
            continue
        Z = np.stack(zs)
        m = Z.mean(axis=0)
        mu[sid] = m
        sigma[sid] = float(np.linalg.norm(Z - m, axis=1).std() + 1e-6)

    mu_answer = sigma_answer = None
    if len(answer_zs) >= MIN_STEP_SUPPORT:
        Za = np.stack(answer_zs)
        mu_answer = Za.mean(axis=0)
        sigma_answer = float(np.linalg.norm(Za - mu_answer, axis=1).std() + 1e-6)

    return DistanceModel(scaler=scaler, pca=pca, mu=mu, sigma=sigma,
                         mu_answer=mu_answer, sigma_answer=sigma_answer, layer=layer)


def _conf(z: np.ndarray, mu: np.ndarray, sigma: float) -> Tuple[float, float]:
    delta = float(np.linalg.norm(z - mu))
    return float(np.exp(-delta / sigma)), delta


def score_example(model: DistanceModel, e: ExampleActs) -> Dict[str, Optional[float]]:
    """Per-step + answer sigma-normalized confidence, plus last-step/cumulative aggregates."""
    per_step_conf: Dict[int, float] = {}
    for sid in e.steps:
        if sid not in model.mu:
            continue
        conf, _ = _conf(model._z(e.step_vec(sid, model.layer)), model.mu[sid], model.sigma[sid])
        per_step_conf[sid] = conf

    answer_conf = None
    if e.answer is not None and model.mu_answer is not None:
        answer_conf, _ = _conf(model._z(e.answer_vec(model.layer)), model.mu_answer, model.sigma_answer)

    last_step_conf = None
    if per_step_conf:
        last_step_conf = per_step_conf[max(per_step_conf)]

    cumulative_conf = float(np.mean(list(per_step_conf.values()))) if per_step_conf else None

    # Example-level confidence: prefer the answer-marker position (closest to the paper's
    # late-step divergence signal); fall back to the last scored step if there's no answer marker.
    final = answer_conf if answer_conf is not None else last_step_conf

    return {
        "final": final, "last_step": last_step_conf, "cumulative": cumulative_conf,
        "answer": answer_conf, "n_steps_scored": len(per_step_conf),
    }


def score_examples(model: DistanceModel, examples: Dict[str, ExampleActs]) -> Dict[str, Dict]:
    return {eid: score_example(model, e) for eid, e in examples.items()}


def evaluate_confidence(scores: Dict[str, Dict], examples: Dict[str, ExampleActs], key: str = "final") -> Dict:
    y, p = [], []
    for eid, s in scores.items():
        if s[key] is None or examples[eid].correct not in (0, 1):
            continue
        y.append(examples[eid].correct)
        p.append(s[key])
    if len(set(y)) < 2:
        return {"auc": float("nan"), "n": len(y)}
    return {"auc": float(roc_auc_score(y, p)), "n": len(y)}


# --------------------------------- calibration ---------------------------------

@dataclass
class PlattCalibrator:
    a: float
    b: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(scores) + self.b)))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Standard Platt scaling: an (effectively) unregularized 1D logistic fit mapping raw
    scores -> calibrated P(correct). C is large (not infinite/unregularized) purely for
    numerical stability with sklearn's solver."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(scores).reshape(-1, 1), labels)
    return PlattCalibrator(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(probs), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = np.asarray(probs), np.asarray(labels)
    return float(np.mean((probs - labels) ** 2))


# --------------------------------- Metric B glue ---------------------------------

def fit_metric_b(X_train: np.ndarray, y_train: np.ndarray, pca_dim: Optional[int] = 128,
                 seed: int = 42) -> Tuple[FittedPredictor, Dict]:
    """Fit predictor.py's classifier on the train split only (no internal holdout -- calib/test
    are the caller's separate splits, matching predictor.fit_deployable's contract)."""
    return fit_deployable(X_train, y_train, pca_dim=pca_dim, seed=seed)
