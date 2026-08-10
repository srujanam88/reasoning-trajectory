"""Visualizations: t-SNE (paper Fig 1a / Fig 4), probe accuracy by layer (Fig 1b), and
correctness-predictor AUC by layer (Fig 2b). All CPU-only -- reads the already-cached
activations.dat/index.parquet, no GPU or new generation needed.
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from .features import ExampleActs
from .probes import ANSWER_LABEL, build_step_identity_matrix


def save_probe_accuracy(acc: Dict, path: str) -> str:
    """Persist the per-class, per-layer probe accuracy dict to JSON so subset re-plots need
    only this small file -- no activations.dat, no recompute. Keys are stringified
    (class label, layer index); load_probe_accuracy reverses it."""
    obj = {str(cls): {str(l): float(a) for l, a in per_layer.items()}
           for cls, per_layer in acc.items()}
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def load_probe_accuracy(path: str) -> Dict:
    """Inverse of save_probe_accuracy: restores int step-ids / 'answer' and int layer keys."""
    raw = json.load(open(path))
    acc = {}
    for cls_s, per_layer in raw.items():
        cls = ANSWER_LABEL if cls_s == ANSWER_LABEL else (int(cls_s) if cls_s.lstrip("-").isdigit() else cls_s)
        acc[cls] = {int(l): float(a) for l, a in per_layer.items()}
    return acc

# A handful of representative layers spanning shallow -> deep, matching the paper's own
# qualitative description ("layer 0" vs "later layers ~11, 21, 31").
DEFAULT_TSNE_LAYERS = [0, 11, 21, 31]


def plot_tsne_grid(
    examples: Dict[str, ExampleActs], layers: List[int], out_path: str,
    perplexity: float = 30.0, seed: int = 42, max_points: int = 3000,
):
    """Fig 1a/4: one t-SNE panel per layer, points colored by step ordinal / answer marker."""
    X, y = build_step_identity_matrix(examples)  # X: [M, n_hidden, d], y: object labels
    if len(y) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_points, replace=False)
        X, y = X[idx], y[idx]

    classes = sorted(set(y.tolist()), key=lambda c: (c == ANSWER_LABEL, c))
    cmap = plt.get_cmap("tab10", len(classes))
    color_of = {c: cmap(i) for i, c in enumerate(classes)}

    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.2))
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        Z = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(
            X[:, layer, :].astype(np.float64)
        )
        for c in classes:
            mask = y == c
            label = "answer" if c == ANSWER_LABEL else f"step {c}"
            ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.6, color=color_of[c], label=label)
        ax.set_title(f"layer {layer}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, markerscale=2)
    fig.suptitle("t-SNE of step-boundary activations across layers")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_probe_accuracy_by_layer(acc: Dict, out_path: str, include_classes=None):
    """Fig 1b: per-class (step 1..K, answer) probe accuracy vs. layer.

    include_classes: optional list of class labels to plot (e.g. [1, 2, 3, 5, 8, "answer"]);
    None plots every class present. Restricting keeps colors distinct and the curves readable.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = sorted(acc.keys(), key=lambda c: (c == ANSWER_LABEL, c))
    if include_classes is not None:
        wanted = {ANSWER_LABEL if str(c).lower() in ("answer", "ans") else c for c in include_classes}
        classes = [c for c in classes if c in wanted]
    cmap = plt.get_cmap("tab10")  # 10 distinct base colors
    for i, c in enumerate(classes):
        per_layer = acc[c]
        layers = sorted(per_layer)
        vals = [per_layer[l] for l in layers]
        label = "answer marker" if c == ANSWER_LABEL else f"step {c}"
        ax.plot(layers, vals, marker="o", markersize=3, color=cmap(i % 10), label=label)
    ax.set_xlabel("layer (hidden_states index)")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Step-identity linear probe accuracy by layer")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_auc_by_layer(per_layer: Dict[int, Dict], out_path: str, paper_peak: float = 0.87):
    """Fig 2b: correctness-predictor AUC vs. layer, peak marked."""
    layers = sorted(per_layer)
    aucs = [per_layer[l]["auc"] for l in layers]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, aucs, marker="o", markersize=3, color="tab:blue", label="trajectory (late_trajectory)")
    valid = [(l, a) for l, a in zip(layers, aucs) if not np.isnan(a)]
    if valid:
        peak_l, peak_a = max(valid, key=lambda t: t[1])
        ax.scatter([peak_l], [peak_a], color="tab:red", zorder=5, label=f"peak L{peak_l}={peak_a:.3f}")
    ax.axhline(paper_peak, color="gray", linestyle="--", linewidth=1, label=f"paper peak {paper_peak}")
    ax.set_xlabel("layer (hidden_states index)")
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Correctness predictor AUC by layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
