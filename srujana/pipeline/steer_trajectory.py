"""Stage 8: trajectory-based steering (paper §5.1).

Unlike Stage 7's one-shot intervention, corrections here can compound (a correction at
step j can change what's written at step j+1 onward), so offline marker positions from
an unsteered run can't be reused past the first intervention -- this needs genuine online
step-boundary detection during generation.

Timing-correct design (preemptive lookahead, matching the paper's own Appendix G method):
decoding is greedy/deterministic, so at each step, BEFORE committing to a token, we do a
short unhooked lookahead to see whether the tentative continuation starts a new "Step k:"
marker. If it does, the *current* position is exactly t(Step k)-1 -- apply the correction
there (if thresholds are exceeded), redo the forward pass with the correction hooked in to
get corrected logits, and commit that token instead; the lookahead tokens are discarded,
they were only a probe. This matches Appendix G's description of the paper's own reference
implementation ("manual token-by-token loop... full growing sequence... reprocessed at
every token") -- a faithful, if slow, behavioral replication, chosen over a faster
retroactive-correction alternative that has a genuine timing bug (by the time a marker is
detected retroactively, its tokens are already committed and causally spent).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

from .features import ExampleActs
from .markers import STEP_RE
from .steering import SteeringHook, hs_idx_to_hook_module


@dataclass
class IdealTrajectory:
    pca: PCA
    mu: Dict[int, np.ndarray]      # per-step mean, in PCA space
    sigma: Dict[int, float]        # per-step dispersion, in PCA space
    layer: int                     # hidden_states index this was fit at


def fit_ideal_trajectory(
    examples: Dict[str, ExampleActs], layer: int, pca_dim: int = 128,
    min_per_step: int = 10, seed: int = 42,
) -> IdealTrajectory:
    """PCA fit on correct-example step-preceding activations at `layer`, pooled across
    step positions into one shared subspace (paper: "extract activations immediately
    preceding each Step token and project them into a low-dimensional subspace via PCA").
    mu_j/sigma_j computed per step ordinal in that projected space.
    """
    rows, step_ids = [], []
    for e in examples.values():
        if e.correct != 1:
            continue
        for step_id, vec in e.steps.items():
            rows.append(vec[layer])
            step_ids.append(step_id)
    X = np.stack(rows).astype(np.float32)
    step_ids = np.array(step_ids)

    pca = PCA(n_components=pca_dim, random_state=seed)
    Z = pca.fit_transform(X)

    mu: Dict[int, np.ndarray] = {}
    sigma: Dict[int, float] = {}
    for j in sorted(set(step_ids.tolist())):
        mask = step_ids == j
        if mask.sum() < min_per_step:
            continue
        zj = Z[mask]
        mu_j = zj.mean(axis=0)
        mu[int(j)] = mu_j
        sigma[int(j)] = float(np.mean(np.linalg.norm(zj - mu_j, axis=1)))
    return IdealTrajectory(pca=pca, mu=mu, sigma=sigma, layer=layer)


def project(traj: IdealTrajectory, h: np.ndarray) -> np.ndarray:
    return traj.pca.transform(h.reshape(1, -1))[0]


def deviation(traj: IdealTrajectory, z: np.ndarray, step_id: int) -> Optional[float]:
    if step_id not in traj.mu:
        return None
    return float(np.linalg.norm(z - traj.mu[step_id]))


def low_rank_correction(
    traj: IdealTrajectory, z: np.ndarray, step_id: int, alpha: float, rank: int = 32
) -> Optional[np.ndarray]:
    """h' = h + alpha * (mu_j - z)[:rank] @ U_r, U_r = top-`rank` principal components
    (traj.pca.components_[:rank], shape [rank, d] -- matches Appendix F's U_r in R^{r x d}
    directly, no transpose needed)."""
    if step_id not in traj.mu:
        return None
    delta_full = traj.mu[step_id] - z
    delta_r = delta_full[:rank]
    U_r = traj.pca.components_[:rank]
    return alpha * (delta_r @ U_r)


def tune_thresholds(
    traj: IdealTrajectory, tune_examples: Dict[str, ExampleActs],
    percentiles: Tuple[int, ...] = (50, 60, 70, 80, 90),
) -> List[Dict]:
    """Offline (CPU, free) shortlist of per-step threshold candidates, using ALREADY
    CACHED (unsteered) activations for correct tune-slice examples -- no regeneration.
    Threshold at the p-th percentile of the CORRECT population's own deviation
    distribution at each step: lower p = stricter (more interventions, more false
    positives on correct trajectories), higher p = more permissive. The caller confirms
    the best candidate with a small number of REAL regeneration passes (not a
    combinatorial grid -- each candidate needs its own generation pass since corrections
    compound)."""
    per_step_delta = defaultdict(list)
    per_step_cum = defaultdict(list)
    for e in tune_examples.values():
        if e.correct != 1:
            continue
        cum = 0.0
        for step_id in sorted(e.steps):
            if step_id not in traj.mu:
                continue
            z = project(traj, e.step_vec(step_id, traj.layer))
            d = deviation(traj, z, step_id)
            per_step_delta[step_id].append(d)
            per_step_cum[step_id].append(cum)
            cum += d

    candidates = []
    for p in percentiles:
        delta_thresh = {j: float(np.percentile(v, p)) for j, v in per_step_delta.items() if v}
        cum_thresh = {j: float(np.percentile(v, p)) for j, v in per_step_cum.items() if v}
        candidates.append({"percentile": p, "delta_thresh": delta_thresh, "cum_thresh": cum_thresh})
    return candidates


@torch.no_grad()
def generate_with_trajectory_steering(
    model, tokenizer, prompt_ids: List[int], traj: IdealTrajectory,
    delta_thresh: Dict[int, float], cum_thresh: Dict[int, float], alpha: float,
    n_hidden: int, device: str, rank: int = 32, max_new_tokens: int = 512,
    lookahead_tokens: int = 8,
) -> Dict:
    """No-cache, full-reprocess-per-token loop with preemptive lookahead detection.
    Slow by design (matches the paper's own reference implementation, Appendix G) --
    intended for small subsamples, not full-scale batch generation.
    """
    layer = traj.layer
    eos_id = tokenizer.eos_token_id
    ids = list(prompt_ids)
    gen_ids: List[int] = []
    step_count = 0
    cum_dev = 0.0
    n_interventions = 0

    def fwd(seq_ids):
        t = torch.tensor([seq_ids], dtype=torch.long, device=device)
        return model(input_ids=t, use_cache=False, output_hidden_states=True)

    for _ in range(max_new_tokens):
        out = fwd(ids)
        cand = int(torch.argmax(out.logits[0, -1]).item())

        # Unhooked lookahead: does the tentative continuation start a new "Step k:"?
        look_ids = [cand]
        cur = list(ids) + [cand]
        for _ in range(lookahead_tokens - 1):
            if cand == eos_id:
                break
            o = fwd(cur)
            nid = int(torch.argmax(o.logits[0, -1]).item())
            look_ids.append(nid)
            cur.append(nid)
        look_text = tokenizer.decode(look_ids, skip_special_tokens=True)
        m = STEP_RE.search(look_text)

        if m is not None and m.start() <= 3:
            step_count += 1
            j = step_count
            h = out.hidden_states[layer][0, -1].float().cpu().numpy()
            z = project(traj, h)
            d = deviation(traj, z, j)
            fire = d is not None and (
                d > delta_thresh.get(j, float("inf")) or cum_dev > cum_thresh.get(j, float("inf"))
            )
            if fire:
                corr = low_rank_correction(traj, z, j, alpha, rank=rank)
                module = hs_idx_to_hook_module(model, layer, n_hidden)
                hook = SteeringHook(module, torch.tensor(corr, dtype=torch.float32), alpha=1.0, position=-1)
                try:
                    out2 = fwd(ids)
                finally:
                    hook.remove()
                cand = int(torch.argmax(out2.logits[0, -1]).item())
                n_interventions += 1
            if d is not None:
                cum_dev += d

        if cand == eos_id:
            break
        gen_ids.append(cand)
        ids.append(cand)

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"gen_ids": gen_ids, "gen_text": gen_text, "n_steps_detected": step_count,
            "n_interventions": n_interventions}
