"""Stage 6: correctness predictor + baselines.

Single-layer logistic regression in PyTorch (Appendix F):
    nn.Linear(d, 1), BCE loss, Adam(weight_decay = 1/C), lr=0.01, batch=32,
    <=1000 epochs, early stopping patience=50 on validation loss.
    C selected via 5-fold StratifiedKFold over {0.001,0.01,0.1,1,10,100}.
    Data split 90/10 stratified. PCA(n_components=128) fit on the train split only.

Feature sets (final-layer activations unless a layer sweep is requested):
    early_step        : concat(step1, step2)                         (~0.63 AUC)
    late_trajectory   : concat(last_step, answer) -> PCA-128         (~0.85 AUC, peak ~L29)
    final_state       : answer -> PCA-128                            (~0.81 AUC)
Baseline:
    step_count        : [n_steps]                                    (~0.649 AUC)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .features import ExampleActs

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ----------------------------- feature construction -----------------------------

def build_features(examples: Dict[str, ExampleActs], kind: str, layer: int):
    """Return (X_raw [M, feat], y [M]) for examples with a known correctness label.

    Feature dimensionality reduction (PCA) is applied later, fit on train only.
    """
    X, y = [], []
    for e in examples.values():
        if e.correct not in (0, 1):
            continue
        steps = sorted(e.steps)
        feat = None
        if kind == "early_step":
            if len(steps) >= 2:
                feat = np.concatenate([e.step_vec(steps[0], layer), e.step_vec(steps[1], layer)])
        elif kind == "late_trajectory":
            if steps and e.answer is not None:
                feat = np.concatenate([e.step_vec(steps[-1], layer), e.answer_vec(layer)])
        elif kind == "final_state":
            if e.answer is not None:
                feat = e.answer_vec(layer)
        elif kind == "step_count":
            feat = np.array([float(e.n_steps)], dtype=np.float32)
        else:
            raise ValueError(f"unknown feature kind {kind}")
        if feat is not None:
            X.append(np.asarray(feat, dtype=np.float32))
            y.append(e.correct)
    if not X:
        return np.empty((0, 0), np.float32), np.empty((0,), int)
    return np.stack(X), np.array(y, dtype=int)


# ----------------------------- torch logistic regression -----------------------------

class _TorchLR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def _train(Xtr, ytr, Xva, yva, C, lr=0.01, batch=32, max_epochs=1000, patience=50, seed=42):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = _TorchLR(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1.0 / C)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    n = len(Xtr_t)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), yva_t).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _select_C(X, y, seed=42):
    """5-fold stratified CV; pick C with best mean validation AUC."""
    best_C, best_auc = 1.0, -1.0
    for C in C_GRID:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for tr, va in skf.split(X, y):
            if len(set(y[va])) < 2:
                continue
            model, _ = _train(X[tr], y[tr], X[va], y[va], C, seed=seed)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(X[va], dtype=torch.float32))).numpy()
            aucs.append(roc_auc_score(y[va], p))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_C = float(np.mean(aucs)), C
    return best_C, best_auc


def fit_eval(
    X: np.ndarray,
    y: np.ndarray,
    pca_dim: Optional[int] = 128,
    seed: int = 42,
) -> Dict:
    """90/10 split, PCA(+scale) fit on train, C via 5-fold CV on train, report test AUC."""
    if len(X) < 10 or len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "n": len(X), "C": None, "note": "insufficient data"}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    if pca_dim and Xtr.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    C, cv_auc = _select_C(Xtr, ytr, seed=seed)
    # Carve a small val set from train for early stopping on the final fit.
    Xt2, Xv2, yt2, yv2 = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=seed)
    model, _ = _train(Xt2, yt2, Xv2, yv2, C, seed=seed)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy()
    return {
        "auc": float(roc_auc_score(yte, p)),
        "cv_auc": cv_auc,
        "C": C,
        "n": len(X),
        "dim": Xtr.shape[1],
    }


def layer_sweep(examples, kind: str, layers: List[int], pca_dim: int = 128, seed: int = 42):
    """Run fit_eval per layer for a feature kind; return {layer: result} and the peak."""
    per_layer = {}
    for L in layers:
        X, y = build_features(examples, kind, L)
        per_layer[L] = fit_eval(X, y, pca_dim=pca_dim, seed=seed)
    valid = {L: r for L, r in per_layer.items() if not np.isnan(r["auc"])}
    peak = max(valid, key=lambda L: valid[L]["auc"]) if valid else None
    return per_layer, peak
