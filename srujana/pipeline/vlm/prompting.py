"""Prompt assembly for the thinking VLM.

The model reasons inside <think>...</think> then answers. We append a step-structure
instruction (pilot decides whether it is honored) and a fixed final-answer marker so the
answer position is anchorable, analogous to the paper's `####`.
"""
from __future__ import annotations

from typing import List

from .datasets import VLMExample

ANSWER_MARK = "Answer:"

# "think" (default): let the model reason naturally inside <think>; we segment that region by
# sentence later, so we do NOT ask for a "Step N:" summary (it would just waste tokens). "marker"
# asks for explicit "Step N:" structure; "paragraph" is free-form. All fix the final-answer marker.
_ANSWER_LINE = f'After you finish reasoning, on a new line write your final answer as "{ANSWER_MARK} <answer>".'
_STEP_INSTRUCTION = {
    "think": _ANSWER_LINE,
    "paragraph": "Think step by step. " + _ANSWER_LINE,
    "marker": ('Show your reasoning as a numbered list where each step begins with '
               '"Step N:" (Step 1:, Step 2:, ...). ' + _ANSWER_LINE),
}


def build_messages(ex: VLMExample, step_mode: str = "think") -> List[dict]:
    """Chat messages for processor.apply_chat_template: image placeholders + question + instruction."""
    if step_mode not in _STEP_INSTRUCTION:
        raise ValueError(f"step_mode must be one of {list(_STEP_INSTRUCTION)}")
    content = [{"type": "image"} for _ in ex.images]
    content.append({"type": "text", "text": f"{ex.question}\n\n{_STEP_INSTRUCTION[step_mode]}"})
    return [{"role": "user", "content": content}]
