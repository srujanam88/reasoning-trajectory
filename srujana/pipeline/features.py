"""Shared per-example activation assembly for Stages 4-6 (CPU-only, reads the memmap).

Turns the flat [N_positions, n_hidden, d] memmap + index.parquet into per-example structures:
    example_id -> {steps: {step_id: [n_hidden, d]}, answer: [n_hidden, d] | None,
                   n_steps: int, correct: int(-1/0/1)}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .extract import load_activations


class ExampleActs:
    __slots__ = ("steps", "answer", "n_steps", "correct")

    def __init__(self, n_steps: int, correct: int):
        self.steps: Dict[int, np.ndarray] = {}
        self.answer: Optional[np.ndarray] = None
        self.n_steps = int(n_steps)
        self.correct = int(correct)

    def step_vec(self, step_id: int, layer: int) -> np.ndarray:
        return self.steps[step_id][layer]

    def answer_vec(self, layer: int) -> Optional[np.ndarray]:
        return None if self.answer is None else self.answer[layer]


def load_examples(run_dir: Path):
    """Return (examples: dict[str, ExampleActs], meta: dict)."""
    acts, index, meta = load_activations(run_dir)
    examples: Dict[str, ExampleActs] = {}
    for r in index.itertuples(index=False):
        e = examples.get(r.example_id)
        if e is None:
            e = ExampleActs(n_steps=r.n_steps, correct=r.correct)
            examples[r.example_id] = e
        vec = np.asarray(acts[r.row], dtype=np.float32)  # [n_hidden, d]
        if r.position_type == "step":
            e.steps[int(r.step_id)] = vec
        else:
            e.answer = vec
    return examples, meta
