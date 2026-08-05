"""Activation steering infra shared by Stage 7 (predictor-gated §4.3) and Stage 9
(length control, §5.2). Stage 8 (§5.1 trajectory steering) has its own module
(`steer_trajectory.py`) since its intervention mechanism is fundamentally different
(online step-boundary detection vs. these two fixed/continuous schemes).

Layer-index convention: "hidden_states index" i (0=embedding, 1..n_hidden-1=decoder
outputs), matching what's stored in activations.dat. IMPORTANT: hidden_states[-1] (the
top slot) is POST-final-norm in HF's output_hidden_states convention -- verified
empirically (a forward hook on model.model.layers[-1] does NOT match hidden_states[-1],
but model.model.norm(hook_output) does). Every other hidden_states[i] for i < n_hidden-1
IS the raw pre-norm decoder-layer output. `hs_idx_to_hook_module` routes around this.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .features import ExampleActs

# Appendix F: LAST = final 5 decoder layers (0-indexed 27-31); MID = 5 layers centered on
# layer 15 (0-indexed 13-17). Shifted +1 into hidden_states-index space (slot 0 = embedding).
LAST_HS = [28, 29, 30, 31, 32]
MID_HS = [14, 15, 16, 17, 18]


def hs_idx_to_hook_module(model, hs_idx: int, n_hidden: int) -> nn.Module:
    """hidden_states[hs_idx] -> the module whose output equals it.

    hs_idx == n_hidden - 1 is the post-final-norm slot (tied to last_hidden_state);
    everything else is the raw pre-norm output of model.model.layers[hs_idx - 1].
    """
    if hs_idx == n_hidden - 1:
        return model.model.norm
    return model.model.layers[hs_idx - 1]


def compute_steering_vectors(
    examples: Dict[str, ExampleActs], layers: List[int]
) -> Dict[int, torch.Tensor]:
    """s^(l) = mean over ALL (example, step_k) pairs of (answer_vec - step_vec) at layer l.

    Pooled across every step ordinal and every example (no correctness filter -- this
    matches the paper's un-gated construction of the steering direction itself; gating
    is a separate decision made later, at intervention time).
    """
    sums: Dict[int, torch.Tensor] = {}
    count = 0
    for e in examples.values():
        if e.answer is None or not e.steps:
            continue
        for step_id, vec in e.steps.items():
            count += 1
            ans = e.answer
            for l in layers:
                diff = torch.tensor(ans[l] - vec[l], dtype=torch.float32)
                sums[l] = sums.get(l, torch.zeros_like(diff)) + diff
    if count == 0:
        raise ValueError("no (step, answer) pairs found to compute steering vectors")
    return {l: s / count for l, s in sums.items()}


class SteeringHook:
    """Adds alpha*vector to a module's output. `position`=None modifies every sequence
    position in whatever tensor comes through (Stage 9, continuous); an int modifies
    only that one column of the CURRENT forward call's tensor (Stage 7, one-shot),
    which is exactly right for a prefill-style call over a known prefix where only the
    final column is the intervention target -- for incremental (cached) calls that only
    ever carry one new position, an int selector of 0 is equivalent to "the only column".
    Starts unarmed; call `.armed = True` to enable, then remove() when done.
    """

    def __init__(self, module: nn.Module, vector: torch.Tensor, alpha: float,
                 position: Optional[int] = None):
        self.armed = True
        self._vector = vector
        self._alpha = alpha
        self._position = position
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.armed:
            return output
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        add = (self._alpha * self._vector).to(dtype=hs.dtype, device=hs.device)
        if self._position is None:
            hs = hs + add
        else:
            pos = self._position if self._position >= 0 else hs.shape[1] + self._position
            hs = hs.clone()
            hs[:, pos, :] = hs[:, pos, :] + add
        return (hs,) + output[1:] if is_tuple else hs

    def remove(self):
        self._handle.remove()


def attach_steering_hooks(
    model, hs_indices: List[int], vectors: Dict[int, torch.Tensor], alpha: float,
    n_hidden: int, position: Optional[int] = None,
) -> List[SteeringHook]:
    hooks = []
    for hs_idx in hs_indices:
        module = hs_idx_to_hook_module(model, hs_idx, n_hidden)
        hooks.append(SteeringHook(module, vectors[hs_idx], alpha, position=position))
    return hooks


@torch.no_grad()
def _greedy_continue(model, tokenizer, input_ids: torch.Tensor, past_key_values,
                      device: str, max_new_tokens: int) -> List[int]:
    """Manual greedy decode loop reusing an existing KV cache, until EOS or budget."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    generated: List[int] = []
    cur = input_ids
    past = past_key_values
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = int(torch.argmax(out.logits[0, -1]).item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


@torch.no_grad()
def generate_with_one_shot_intervention(
    model, tokenizer, prompt_ids: List[int], known_gen_ids: List[int],
    intervene_at_rel: int, hs_indices: List[int], vectors: Dict[int, torch.Tensor],
    alpha: float, n_hidden: int, device: str, max_new_tokens: int = 512,
) -> Dict:
    """Stage 7 (§4.3): regenerate a single example with a ONE-TIME steering intervention
    at the position immediately preceding the answer marker.

    `known_gen_ids` is the ORIGINAL (unsteered) generation's token ids; `intervene_at_rel`
    is the 0-indexed position within `known_gen_ids` of `t(term)-1` (from re-parsing that
    original generation with markers.parse_markers). Because decoding up to that point is
    deterministic/greedy and unperturbed, replaying `known_gen_ids[:intervene_at_rel+1]`
    as the prefix reproduces the original run's tokens exactly -- no re-decoding needed.
    The hook fires once, on the single big prefill forward over prompt+prefix, then is
    removed and generation continues unhooked to completion.
    """
    prefix = list(prompt_ids) + list(known_gen_ids[: intervene_at_rel + 1])
    prefix_t = torch.tensor([prefix], dtype=torch.long, device=device)

    hooks = attach_steering_hooks(model, hs_indices, vectors, alpha, n_hidden,
                                   position=len(prefix) - 1)
    try:
        out = model(input_ids=prefix_t, use_cache=True)
    finally:
        for h in hooks:
            h.remove()

    next_id = int(torch.argmax(out.logits[0, -1]).item())
    eos_id = tokenizer.eos_token_id
    gen_ids = list(known_gen_ids[: intervene_at_rel + 1])
    past = out.past_key_values
    if next_id != eos_id:
        gen_ids.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        gen_ids += _greedy_continue(model, tokenizer, cur, past, device,
                                     max_new_tokens - len(gen_ids))
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"gen_ids": gen_ids, "gen_text": gen_text}


@torch.no_grad()
def generate_with_continuous_intervention(
    model, tokenizer, prompt_ids: List[int], hs_indices: List[int],
    vectors: Dict[int, torch.Tensor], alpha: float, n_hidden: int, device: str,
    steer_prompt: bool = False, max_new_tokens: int = 512,
) -> Dict:
    """Stage 9 (§5.2): same steering vectors/layers as Stage 7, but the hook stays armed
    for every generated token (SHORTEN/PROLONG length control), not a single position.

    `steer_prompt=False` (default) keeps the hook unarmed during the prompt prefill --
    steering only the model's own generated reasoning, not the question text itself.
    The paper doesn't specify this choice explicitly; exposed as a flag.
    """
    hooks = attach_steering_hooks(model, hs_indices, vectors, alpha, n_hidden, position=None)
    try:
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for h in hooks:
            h.armed = steer_prompt
        out = model(input_ids=prompt_t, use_cache=True)
        for h in hooks:
            h.armed = True

        eos_id = tokenizer.eos_token_id
        next_id = int(torch.argmax(out.logits[0, -1]).item())
        gen_ids: List[int] = []
        past = out.past_key_values
        if next_id != eos_id:
            gen_ids.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
            for _ in range(max_new_tokens - 1):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(torch.argmax(o.logits[0, -1]).item())
                if nid == eos_id:
                    break
                gen_ids.append(nid)
                cur = torch.tensor([[nid]], dtype=torch.long, device=device)
    finally:
        for h in hooks:
            h.remove()

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit_budget = len(gen_ids) >= max_new_tokens
    return {"gen_ids": gen_ids, "gen_text": gen_text, "hit_budget": hit_budget}
