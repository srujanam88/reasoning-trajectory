"""Per-dataset correctness labeling for the VLM outputs.

The final answer is taken from after the last `Answer:` marker (fallback: after `</think>`).
- CV-Bench: gold like "(C)" -> match extracted choice letter.
- MMMU: MC gold is a letter ("C"); open -> normalized value.
- MathVista: MC gold is the choice *text* (map model letter->choice); free_form -> numeric/text.
First-pass rules; refine against real outputs during the Phase-0 pilot.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..label import answers_match, normalize_answer
from .prompting import ANSWER_MARK

_NUM = r"-?\d+(?:[\s,]\d+)*(?:\.\d+)?"


def answer_span(gen_text: str) -> str:
    idx = gen_text.rfind(ANSWER_MARK)
    if idx >= 0:
        return gen_text[idx + len(ANSWER_MARK):].strip()
    t = gen_text.rfind("</think>")
    if t >= 0:
        return gen_text[t + len("</think>"):].strip()
    return gen_text.strip()


def _extract_letter(text: str, n_choices: int = 26) -> Optional[str]:
    m = re.search(r"\(([A-Za-z])\)", text)          # "(C)"
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Z])\b", text.strip()[:8])  # a leading standalone capital
    if m and (ord(m.group(1)) - 65) < n_choices:
        return m.group(1)
    return None


def _extract_number(text: str) -> Optional[str]:
    nums = re.findall(rf"({_NUM})", text)
    return nums[0].replace(" ", "").replace(",", "") if nums else None


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def label_example(dataset: str, gen_text: str, gold: str, choices: Optional[List[str]],
                  question_type: str, meta: Optional[dict] = None) -> bool:
    meta = meta or {}
    span = answer_span(gen_text)

    if dataset == "cvbench":
        gl = re.sub(r"[()\s]", "", gold).upper()
        return _extract_letter(span, len(choices) if choices else 26) == gl

    if dataset == "mmmu":
        if str(question_type).lower().startswith("open"):
            return answers_match(_extract_number(span), gold) or _norm_text(span) == _norm_text(gold)
        gl = re.sub(r"[()\s]", "", gold).upper()
        return _extract_letter(span, len(choices) if choices else 26) == gl

    if dataset == "mathvista":
        if question_type == "multiple-choice" and choices:
            pl = _extract_letter(span, len(choices))
            pred_choice = choices[ord(pl) - 65] if pl and (ord(pl) - 65) < len(choices) else span
            return _norm_text(pred_choice) == _norm_text(gold)
        # free_form
        if meta.get("answer_type") in ("integer", "float"):
            return answers_match(_extract_number(span), gold)
        return answers_match(_extract_number(span), gold) or _norm_text(span) == _norm_text(gold)

    raise KeyError(f"unknown dataset '{dataset}'")
