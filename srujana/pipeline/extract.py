"""Two-pass hidden-state extraction to a memmap (Stage 2).

Pass 1 (generation) happened in Stage 1. Here we do Pass 2: a single teacher-forced forward
over prompt+generated tokens with ``output_hidden_states=True, use_cache=False``, and cache the
activation at each marker's preceding position across ALL hidden layers.

Storage:
  activations.dat  : float16 memmap, shape [N_positions, n_hidden, hidden_size]
  index.parquet    : one row per cached position (row, example_id, position_type, step_id,
                     token_index, n_steps, correct[filled in Stage 3])
  meta.json        : {N, n_hidden, hidden_size, model, ...}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .generate import GenResult
from .markers import MarkerResult, parse_markers


def _iter_positions(mk: MarkerResult, seq_len: int):
    """Yield (position_type, step_id, abs_prec_pos) for valid cached positions."""
    for s in mk.steps:
        if 0 <= s.abs_prec_pos < seq_len:
            yield ("step", s.step_id, s.abs_prec_pos)
    if mk.answer is not None and 0 <= mk.answer.abs_prec_pos < seq_len:
        yield ("answer", 0, mk.answer.abs_prec_pos)


@torch.no_grad()
def extract(
    model,
    tokenizer,
    gens: List[GenResult],
    device: str,
    out_dir: Path,
    model_name: str,
    checkpoint_every: int = 100,
) -> Dict:
    """Run Pass 2 over all examples and write the activation memmap + index.

    Returns a summary dict including the teacher-forcing argmax match rate (should be ~1.0).
    """
    n_hidden = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size

    # First parse markers and count total positions so we can size the memmap.
    markers: List[MarkerResult] = []
    total = 0
    for g in gens:
        seq_len = g.prompt_len + len(g.gen_ids)
        mk = parse_markers(g.example_id, g.gen_ids, tokenizer, g.prompt_len)
        markers.append(mk)
        total += sum(1 for _ in _iter_positions(mk, seq_len))

    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations.dat"
    acts = np.memmap(act_path, dtype=np.float16, mode="w+", shape=(total, n_hidden, hidden_size))

    index_rows: List[Dict] = []
    row = 0
    tf_match, tf_total = 0, 0

    for ei, (g, mk) in enumerate(zip(gens, markers)):
        full_ids = g.prompt_ids + g.gen_ids
        seq_len = len(full_ids)
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(inp, output_hidden_states=True, use_cache=False)

        # Teacher-forcing sanity: argmax(logits[q-1]) == full_ids[q] over generated positions.
        logits = out.logits[0]  # [seq, vocab]
        for q in range(g.prompt_len, seq_len):
            pred = int(torch.argmax(logits[q - 1]).item())
            tf_total += 1
            tf_match += int(pred == full_ids[q])

        hs = out.hidden_states  # tuple length n_hidden, each [1, seq, hidden]
        for ptype, step_id, p in _iter_positions(mk, seq_len):
            vec = torch.stack([hs[l][0, p] for l in range(n_hidden)])  # [n_hidden, hidden]
            acts[row] = vec.float().cpu().numpy().astype(np.float16)
            index_rows.append(
                {
                    "row": row,
                    "example_id": g.example_id,
                    "position_type": ptype,
                    "step_id": step_id,
                    "token_index": p,
                    "n_steps": mk.n_steps,
                    "correct": -1,  # filled in Stage 3
                }
            )
            row += 1

        if checkpoint_every and (ei + 1) % checkpoint_every == 0:
            acts.flush()

    acts.flush()

    import pandas as pd

    df = pd.DataFrame(index_rows)
    df.to_parquet(out_dir / "index.parquet")

    meta = {
        "N": total,
        "n_hidden": n_hidden,
        "hidden_size": hidden_size,
        "model": model_name,
        "dtype": "float16",
        "tf_match_rate": (tf_match / tf_total) if tf_total else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_slim(out_dir: Path, layers=None, out_name: str = "slim_acts.npz") -> Path:
    """Write a compact npz keeping only `layers` (default: final layer) of the full memmap.

    Run this on the box that has activations.dat. The result (~45 MB for one layer / 5.5k
    positions vs ~1.5 GB for the full 33-layer memmap) is enough for distance analysis and any
    final-layer probe/predictor, and is loaded transparently by features.load_examples.
    """
    acts, index, meta = load_activations(out_dir)
    if layers is None:
        layers = [meta["n_hidden"] - 1]
    layers = [int(l) for l in layers]
    sub = np.asarray(acts[:, layers, :], dtype=np.float16)  # [N, k, d]
    out_path = Path(out_dir) / out_name
    np.savez(
        out_path, acts=sub, layers=np.array(layers, dtype=np.int64),
        n_hidden=np.int64(meta["n_hidden"]), hidden_size=np.int64(meta["hidden_size"]),
    )
    return out_path


def load_activations(out_dir: Path):
    """Open the memmap read-only along with meta + index."""
    import pandas as pd

    meta = json.loads((out_dir / "meta.json").read_text())
    acts = np.memmap(
        out_dir / "activations.dat",
        dtype=np.dtype(meta["dtype"]),
        mode="r",
        shape=(meta["N"], meta["n_hidden"], meta["hidden_size"]),
    )
    index = pd.read_parquet(out_dir / "index.parquet")
    return acts, index, meta
