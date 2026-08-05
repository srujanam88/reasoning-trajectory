"""Model registry, device/dtype selection, paths.

Single-device by design (local Mac MPS/CPU for the 1B test model; single RunPod GPU
for the 8B models). Model is selectable by short name via ``--model``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Short name -> HuggingFace id.
MODELS = {
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",   # local plumbing test
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",   # first real target
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",                     # base (later)
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # later
}

# Repo root and output locations.
PKG_DIR = Path(__file__).resolve().parent
SRUJANA_DIR = PKG_DIR.parent
OUTPUTS_DIR = SRUJANA_DIR / "outputs"


def run_dir(model: str, dataset: str = "gsm8k") -> Path:
    """Per-(model, dataset) output directory; created on demand."""
    d = OUTPUTS_DIR / f"{model}__{dataset}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def steer_run_dir(model: str, dataset: str, tag: str) -> Path:
    """Per-(model, dataset, tag) steering-experiment directory (Stages 7-9).

    Kept under run_dir(model, dataset)/steer/tag so regenerated outputs never overwrite
    the original gen.jsonl / activations.dat / index.parquet.
    """
    d = run_dir(model, dataset) / "steer" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    # bf16 on real GPU (per paper_details.md); fp32 elsewhere for numerical safety on MPS/CPU.
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """Load a model+tokenizer from the registry onto a single device.

    Tokenizer uses left padding (required for batched greedy generation) and falls back
    to eos as pad token (Llama has no dedicated pad token).
    """
    if model not in MODELS:
        raise KeyError(f"Unknown model '{model}'. Known: {list(MODELS)}")
    hf_id = MODELS[model]
    device = device or get_device()
    dtype = dtype or get_dtype(device)

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    net = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    net = net.to(device).eval()
    return net, tok, device
