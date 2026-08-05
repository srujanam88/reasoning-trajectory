"""Correctness labeling (Stage 3).

Answer extraction ported (lean) from the reference impl's
`src/features/span_detection.extract_answer_after_hash`, plus GSM8K-style numeric
normalization. Kept self-contained so srujana/ has no import coupling to src/.
"""
from __future__ import annotations

import re
from typing import Optional

_CURRENCY = r"[\$£€¥₹]"
_NUM = r"-?\d+(?:[\s,]\d+)*(?:\.\d+)?"


def _last_number(txt: str) -> Optional[str]:
    nums = re.findall(rf"({_NUM})", txt)
    if nums:
        return nums[-1].replace(" ", "").replace(",", "")
    return None


def extract_answer_after_hash(text: str) -> Optional[str]:
    """Extract the numeric answer after the first '####'.

    Handles currency, commas, spaces-between-digits, and 'a op b = c' chains
    (recursing on '=' when the first number is followed by an operator). Falls back to the
    last number in the whole text if no '####' or extraction fails.
    """
    if not text:
        return None
    if "####" not in text:
        return _last_number(text)

    after = text.split("####", 1)[1].strip()
    after = re.sub(rf"^{_CURRENCY}+\s*", "", after)

    m = re.search(rf"({_NUM})", after)
    if not m:
        return _last_number(text)

    tail = after[m.end():].lstrip()
    if tail and tail[0] in "+-*/":
        remaining = after
        while "=" in remaining:
            rhs = remaining.split("=", 1)[1].strip()
            rhs = re.sub(rf"^{_CURRENCY}+\s*", "", rhs)
            rm = re.search(rf"({_NUM})", rhs)
            if not rm:
                break
            rtail = rhs[rm.end():].lstrip()
            if not rtail or rtail[0] not in "+-*/":
                return rm.group(1).replace(" ", "").replace(",", "")
            remaining = remaining.split("=", 1)[1]
        return _last_number(text)

    return m.group(1).replace(" ", "").replace(",", "")


def normalize_answer(a: Optional[str]) -> Optional[str]:
    if a is None:
        return None
    s = str(a).strip()
    s = re.sub(_CURRENCY, "", s)
    s = s.replace(",", "").replace(" ", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)
