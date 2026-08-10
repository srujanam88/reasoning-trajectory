"""Shared per-example activation assembly for Stages 4-6 (CPU-only).

Turns the flat [N_positions, n_hidden, d] memmap + index.parquet into per-example structures:
    example_id -> {steps: {step_id: [n_hidden, d]}, answer: [n_hidden, d] | None,
                   n_steps: int, correct: int(-1/0/1)}

Also supports a **slim** artifact (`slim_acts.npz`, produced by extract.export_slim) that keeps
only a subset of layers (e.g. just the final layer) so distance / final-layer analysis can run
without syncing the full ~1.5 GB memmap. Callers still pass absolute hidden-state layer indices
(e.g. layer = n_hidden-1); ExampleActs translates them to the slim array's rows transparently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .extract import load_activations


class ExampleActs:
    __slots__ = ("steps", "answer", "n_steps", "correct", "_layers")

    def __init__(self, n_steps: int, correct: int, layers: Optional[List[int]] = None):
        self.steps: Dict[int, np.ndarray] = {}
        self.answer: Optional[np.ndarray] = None
        self.n_steps = int(n_steps)
        self.correct = int(correct)
        # None => stored arrays are full-height and indexable by absolute layer.
        # Otherwise a list of the absolute hidden-state indices present, in row order.
        self._layers = layers

    def _row(self, layer: int) -> int:
        if self._layers is None:
            return layer
        try:
            return self._layers.index(layer)
        except ValueError:
            raise KeyError(
                f"layer {layer} not in slim export (kept layers: {self._layers}). "
                f"Re-export with this layer, or use the full activations.dat."
            )

    def step_vec(self, step_id: int, layer: int) -> np.ndarray:
        return self.steps[step_id][self._row(layer)]

    def answer_vec(self, layer: int) -> Optional[np.ndarray]:
        return None if self.answer is None else self.answer[self._row(layer)]


def _assemble(index, get_vec, layers: Optional[List[int]]) -> Dict[str, ExampleActs]:
    examples: Dict[str, ExampleActs] = {}
    for r in index.itertuples(index=False):
        e = examples.get(r.example_id)
        if e is None:
            e = ExampleActs(n_steps=r.n_steps, correct=r.correct, layers=layers)
            examples[r.example_id] = e
        vec = np.asarray(get_vec(r.row), dtype=np.float32)
        if r.position_type == "step":
            e.steps[int(r.step_id)] = vec
        else:
            e.answer = vec
    return examples


def load_examples(run_dir: Path, slim: Optional[bool] = None):
    """Return (examples: dict[str, ExampleActs], meta: dict).

    slim=None (default): use the slim npz if the full memmap is absent but slim exists;
    slim=True/False: force. meta keeps the true full n_hidden so absolute-layer args still work.
    """
    run_dir = Path(run_dir)
    act_path = run_dir / "activations.dat"
    slim_path = run_dir / "slim_acts.npz"
    use_slim = slim if slim is not None else (not act_path.exists() and slim_path.exists())

    if use_slim:
        import pandas as pd

        z = np.load(slim_path)
        acts = z["acts"]  # [N, k, d]
        layers = [int(x) for x in z["layers"]]
        index = pd.read_parquet(run_dir / "index.parquet")
        meta = json.loads((run_dir / "meta.json").read_text())
        meta = {**meta, "n_hidden": int(z["n_hidden"]), "hidden_size": int(z["hidden_size"]),
                "slim_layers": layers}
        return _assemble(index, lambda row: acts[row], layers), meta

    acts, index, meta = load_activations(run_dir)
    return _assemble(index, lambda row: acts[row], None), meta
