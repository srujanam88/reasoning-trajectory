"""Stage 6 baseline: logit-lens features at step/answer boundaries.

Paper (Table 3): "Logit-lens features at step boundaries (entropy, answer-marker
token rank, and top-1 probability) achieve a best AUC of 0.765 +/- 0.027." We
mirror late_trajectory's two boundary positions (last step, answer) for an
apples-to-apples comparison, computing the 3 scalars at each -> a 6-dim feature
vector per example.

"Logit lens": apply the model's FINAL RMSNorm + lm_head to an *intermediate*-
layer hidden state (never trained for this, but a standard interpretability
probe) to see what the model "would predict" from that layer. This reuses the
hidden states already cached in Stage 2 -- no new generation required, just an
offline unembed, so it can run on CPU per the GPU-budget discipline (PLAN.md S7).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .features import ExampleActs
from .prompting import FINAL_ANSWER_MARK


def answer_marker_token_id(tokenizer) -> int:
    """Token id of the FIRST token of '####' under this tokenizer.

    Empirically this is the actual observed token at essentially every answer
    marker (checked: 984/985 on Llama-3.1-8B-Instruct train generations), so a
    single fixed id is a safe stand-in for the per-example marker token.
    """
    ids = tokenizer.encode(FINAL_ANSWER_MARK, add_special_tokens=False)
    return int(ids[0])


@torch.no_grad()
def _logit_lens_scalars(h: torch.Tensor, model, marker_token_id: int, chunk: int = 512):
    """h: [M, d] float32 stored activations at one layer -> (entropy, top1_prob, rank), each [M]."""
    norm = model.model.norm
    head = model.lm_head
    dtype = next(head.parameters()).dtype
    device = next(head.parameters()).device

    ent, top1, rank = [], [], []
    for i in range(0, h.shape[0], chunk):
        x = h[i : i + chunk].to(device=device, dtype=dtype)
        logits = head(norm(x)).float()  # [b, vocab]
        logp = torch.log_softmax(logits, dim=-1)
        ent.append((-(logp.exp() * logp).sum(-1)).cpu())
        top1.append(logp.max(-1).values.exp().cpu())
        marker_logit = logits[:, marker_token_id].unsqueeze(-1)
        rank.append((logits > marker_logit).sum(-1).float().cpu())
    return torch.cat(ent).numpy(), torch.cat(top1).numpy(), torch.cat(rank).numpy()


def build_logit_lens_features(
    examples: Dict[str, ExampleActs], layer: int, model, marker_token_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X [M, 6], y [M]): [entropy, top1_prob, marker_rank] at (last-step, answer)."""
    vecs_step, vecs_ans, ys = [], [], []
    for e in examples.values():
        if e.correct not in (0, 1):
            continue
        steps = sorted(e.steps)
        if not steps or e.answer is None:
            continue
        vecs_step.append(e.step_vec(steps[-1], layer))
        vecs_ans.append(e.answer_vec(layer))
        ys.append(e.correct)
    if not ys:
        return np.empty((0, 6), np.float32), np.empty((0,), int)

    Hs = torch.tensor(np.stack(vecs_step), dtype=torch.float32)
    Ha = torch.tensor(np.stack(vecs_ans), dtype=torch.float32)
    es, ts, rs = _logit_lens_scalars(Hs, model, marker_token_id)
    ea, ta, ra = _logit_lens_scalars(Ha, model, marker_token_id)
    X = np.stack([es, ts, rs, ea, ta, ra], axis=1).astype(np.float32)
    return X, np.array(ys, dtype=int)


def logit_lens_layer_sweep(examples, layers, model, marker_token_id, fit_eval_fn, seed: int = 42):
    """Per-layer fit_eval using logit-lens scalars (no PCA -- already 6-dim)."""
    per_layer = {}
    for L in layers:
        X, y = build_logit_lens_features(examples, L, model, marker_token_id)
        per_layer[L] = fit_eval_fn(X, y, pca_dim=None, seed=seed)
    valid = {L: r for L, r in per_layer.items() if not np.isnan(r["auc"])}
    peak = max(valid, key=lambda L: valid[L]["auc"]) if valid else None
    return per_layer, peak
