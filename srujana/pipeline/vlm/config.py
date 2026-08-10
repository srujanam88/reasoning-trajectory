"""VLM model registry + loader. Reuses device/dtype/run_dir from the parent config."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .. import config as base

VLM_MODELS = {
    "qwen3-vl-8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",  # 36 layers, hidden 4096 -> n_hidden 37
    "qwen3-vl-8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",  # same backbone, no forced <think>
}


def load_vlm(
    model: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None
) -> Tuple[object, object, str]:
    """Load a VLM + its processor onto a single device (bf16 on CUDA)."""
    if model not in VLM_MODELS:
        raise KeyError(f"Unknown VLM '{model}'. Known: {list(VLM_MODELS)}")
    hf_id = VLM_MODELS[model]
    device = device or base.get_device()
    dtype = dtype or base.get_dtype(device)

    processor = AutoProcessor.from_pretrained(hf_id)
    net = AutoModelForImageTextToText.from_pretrained(hf_id, torch_dtype=dtype)
    net = net.to(device).eval()
    return net, processor, device


def backbone_dims(model_net) -> Tuple[int, int]:
    """(n_hidden, hidden_size) for the LLM backbone: n_hidden = num_hidden_layers + 1."""
    cfg = model_net.config
    tc = getattr(cfg, "text_config", cfg)
    return tc.num_hidden_layers + 1, tc.hidden_size


# Convenience re-exports so callers can use vlm.config for everything.
run_dir = base.run_dir
get_device = base.get_device
