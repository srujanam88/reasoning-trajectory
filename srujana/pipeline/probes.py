"""Stage 4: step-identity linear probes + cross-model transfer.

Per (step class x layer), a binary one-vs-rest logistic regression predicts whether an
activation belongs to that step. Hyperparameters per Appendix F:
    LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs',
                       penalty='l2', C=1.0), 80/20 stratified split.
Labels are the step ordinal (1..K) plus 'answer' for the final-answer-marker position.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .features import ExampleActs

ANSWER_LABEL = "answer"


def build_step_identity_matrix(examples: Dict[str, ExampleActs]):
    """Stack all step/answer activations. Returns X [M, n_hidden, d], y (object labels)."""
    rows, labels = [], []
    for e in examples.values():
        for sid, vec in e.steps.items():
            rows.append(vec)
            labels.append(sid)
        if e.answer is not None:
            rows.append(e.answer)
            labels.append(ANSWER_LABEL)
    X = np.stack(rows).astype(np.float32)  # [M, n_hidden, d]
    y = np.array(labels, dtype=object)
    return X, y


def _fit_layer(Xl, yb, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        Xl, yb, test_size=0.2, stratify=yb, random_state=seed
    )
    clf = LogisticRegression(  # penalty defaults to l2 (Appendix F)
        max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0
    )
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte)), clf


def run_step_probes(
    X: np.ndarray,
    y: np.ndarray,
    layers: Optional[List[int]] = None,
    min_per_class: int = 10,
    shuffle_labels: bool = False,
    seed: int = 42,
) -> Dict:
    """Train per-class, per-layer probes. Returns {'acc': {label: {layer: acc}}, 'classes', 'layers'}."""
    n_hidden = X.shape[1]
    layers = list(range(n_hidden)) if layers is None else layers

    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = y.copy()
        rng.shuffle(y)

    classes = sorted(set(y.tolist()), key=lambda c: (c == ANSWER_LABEL, c))
    acc: Dict[object, Dict[int, float]] = {}
    for cls in classes:
        yb = (y == cls).astype(int)
        if yb.sum() < min_per_class or (len(yb) - yb.sum()) < min_per_class:
            continue
        acc[cls] = {}
        for layer in layers:
            Xl = X[:, layer, :]
            acc[cls][layer], _ = _fit_layer(Xl, yb, seed)
    return {"acc": acc, "classes": list(acc.keys()), "layers": layers}


def cross_model_transfer(
    X_src, y_src, X_tgt, y_tgt, layers: Optional[List[int]] = None, min_per_class: int = 10
) -> Dict:
    """Train probe on source model activations, evaluate on target model. Per class x layer."""
    n_hidden = X_src.shape[1]
    layers = list(range(n_hidden)) if layers is None else layers
    classes = sorted(set(y_src.tolist()) & set(y_tgt.tolist()),
                     key=lambda c: (c == ANSWER_LABEL, c))
    out: Dict[object, Dict[int, float]] = {}
    for cls in classes:
        ys = (y_src == cls).astype(int)
        yt = (y_tgt == cls).astype(int)
        if ys.sum() < min_per_class or yt.sum() < min_per_class:
            continue
        out[cls] = {}
        for layer in layers:
            clf = LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs", penalty="l2", C=1.0
            )
            clf.fit(X_src[:, layer, :], ys)
            out[cls][layer] = float(clf.score(X_tgt[:, layer, :], yt))
    return out


def best_layer_summary(acc: Dict) -> Dict:
    """label -> (best_layer, best_acc, mean_acc_over_layers)."""
    summary = {}
    for label, per_layer in acc.items():
        if not per_layer:
            continue
        best_layer = max(per_layer, key=per_layer.get)
        vals = list(per_layer.values())
        summary[label] = (best_layer, per_layer[best_layer], float(np.mean(vals)))
    return summary
