"""Offset-based Step / #### marker parsing (Stage 1).

We locate marker *token positions* from the TRUE generated token ids, never by
re-encoding decoded text (cf. reference impl's `detect_dp2_index`) and never by a
hardcoded token id (cf. reference `STEP_TOKEN_ID = 8468`).

Method:
  1. Build an exact char->token map from the real tokens: ``offsets[i] = len(decode(ids[:i]))``.
     This is prefix-additive by construction, so it is robust to byte-BPE space handling.
  2. Locate ``Step N:`` and ``####`` by regex on the canonical ``decode(ids)`` text (char offsets).
  3. Map each marker's char offset to the token index whose span contains it.

The cached activation position is the token *preceding* the marker: paper's
``h_{t(Step k)-1}`` and ``h_{t(term)-1}``. In absolute (full-sequence) coordinates that is
``prompt_len + gen_tok_index - 1``.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import List, Optional

STEP_RE = re.compile(r"Step\s*(\d+)\s*:")
# "Step" as its own token in the Llama-3 tokenizer (used only for a cross-check diagnostic).
LLAMA_STEP_TOKEN_ID = 8468


@dataclass
class MarkerPos:
    kind: str            # "step" or "answer"
    step_id: int         # 1-based ordinal (0 for answer)
    parsed_num: Optional[int]  # number literally written in "Step N:" (None for answer)
    char: int            # char offset of marker start in decode(gen_ids)
    gen_tok_index: int   # index of marker token within generated tokens
    abs_marker_pos: int  # prompt_len + gen_tok_index
    abs_prec_pos: int    # abs_marker_pos - 1  (the cached activation position)


@dataclass
class MarkerResult:
    example_id: str
    prompt_len: int
    gen_len: int
    full_text: str
    steps: List[MarkerPos] = field(default_factory=list)
    answer: Optional[MarkerPos] = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def _char_offsets(gen_ids: List[int], tokenizer) -> List[int]:
    """offsets[i] = length in chars of decode(gen_ids[:i]); len == len(gen_ids)+1."""
    offsets = [0]
    for i in range(1, len(gen_ids) + 1):
        offsets.append(len(tokenizer.decode(gen_ids[:i], skip_special_tokens=False)))
    return offsets


def _char_to_tok(offsets: List[int], char: int) -> int:
    """Token index whose span [offsets[j], offsets[j+1]) contains `char`."""
    j = bisect.bisect_right(offsets, char) - 1
    return max(0, min(j, len(offsets) - 2))


def parse_markers(
    example_id: str,
    gen_ids: List[int],
    tokenizer,
    prompt_len: int,
) -> MarkerResult:
    """Parse Step/#### markers from a single example's generated token ids."""
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    offsets = _char_offsets(gen_ids, tokenizer)

    res = MarkerResult(
        example_id=example_id,
        prompt_len=prompt_len,
        gen_len=len(gen_ids),
        full_text=full_text,
    )

    # Answer marker: first "####".
    ans_char = full_text.find("####")
    if ans_char >= 0:
        j = _char_to_tok(offsets, ans_char)
        abs_marker = prompt_len + j
        res.answer = MarkerPos(
            kind="answer", step_id=0, parsed_num=None, char=ans_char,
            gen_tok_index=j, abs_marker_pos=abs_marker, abs_prec_pos=abs_marker - 1,
        )

    # Step markers, ordered; only those in the reasoning region (before the answer marker).
    ordinal = 0
    for m in STEP_RE.finditer(full_text):
        if ans_char >= 0 and m.start() >= ans_char:
            break
        ordinal += 1
        j = _char_to_tok(offsets, m.start())
        abs_marker = prompt_len + j
        res.steps.append(
            MarkerPos(
                kind="step", step_id=ordinal, parsed_num=int(m.group(1)), char=m.start(),
                gen_tok_index=j, abs_marker_pos=abs_marker, abs_prec_pos=abs_marker - 1,
            )
        )
    return res


def step_positions_by_token_id(
    gen_ids: List[int], prompt_len: int, step_token_id: int = LLAMA_STEP_TOKEN_ID
) -> List[int]:
    """Cross-check helper: absolute preceding positions found by matching a step token id.

    Only meaningful for Llama-3 vocab. Returns abs_prec_pos list.
    """
    return [prompt_len + i - 1 for i, tid in enumerate(gen_ids) if tid == step_token_id]
