"""Offset-based step/answer marker parsing for the VLM's generated reasoning.

Reuses the text pipeline's exact-offset machinery (char->token map on the TRUE generated
tokens). Step modes:
  - "think"     : (DEFAULT for Qwen3-VL-Thinking) the prompt asks the model to label its
                  reasoning "Step N:" INSIDE <think>...</think>; steps are the `Step N:`
                  occurrences in that region. If the model doesn't comply (no markers found),
                  falls back to sentence-boundary segmentation of the same region. The
                  post-</think> "Step N:" block (if any) is a summary, not reasoning, and is
                  excluded. The answer marker is the end of </think> (reasoning complete = t(term)).
  - "marker"    : find `Step N:` occurrences across the whole response (paper-faithful, if the
                  model complies).
  - "paragraph" : segment the reasoning region by blank-line paragraphs.
For marker/paragraph the answer marker is `Answer:` (falls back to `</think>`). Cached
activation is the token PRECEDING each marker, exactly as in the text pipeline.
"""
from __future__ import annotations

import re
from typing import Optional

from ..markers import STEP_RE, MarkerPos, MarkerResult, _char_offsets, _char_to_tok

PARA_RE = re.compile(r"\n[ \t]*\n")
# Sentence boundary: end punctuation + space + capital/digit (won't split "10.5").
SENT_RE = re.compile(r"[.!?]+[\)\"']?\s+(?=[A-Z0-9])")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _make_pos(kind, step_id, parsed_num, char, offsets, prompt_len):
    j = _char_to_tok(offsets, char)
    abs_marker = prompt_len + j
    return MarkerPos(kind=kind, step_id=step_id, parsed_num=parsed_num, char=char,
                     gen_tok_index=j, abs_marker_pos=abs_marker, abs_prec_pos=abs_marker - 1)


def parse_markers(
    example_id: str,
    gen_ids,
    tokenizer,
    prompt_len: int,
    step_mode: str = "think",
    answer_mark: str = "Answer:",
) -> MarkerResult:
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    offsets = _char_offsets(gen_ids, tokenizer)
    res = MarkerResult(example_id=example_id, prompt_len=prompt_len, gen_len=len(gen_ids),
                       full_text=full_text)

    if step_mode == "think":
        # Reasoning region = inside <think>...</think>. The <think> open tag is usually part of
        # the prompt (not generated), so default region_start = 0.
        topen = full_text.find(THINK_OPEN)
        region_start = (topen + len(THINK_OPEN)) if topen >= 0 else 0
        tclose = full_text.find(THINK_CLOSE, region_start)
        if tclose >= 0:
            region_end = tclose
            res.answer = _make_pos("answer", 0, None, tclose, offsets, prompt_len)  # t(term)=end of think
        else:
            # No </think>: fall back to Answer: marker, else end of text.
            a = full_text.find(answer_mark)
            region_end = a if a >= 0 else len(full_text)
            if a >= 0:
                res.answer = _make_pos("answer", 0, None, a, offsets, prompt_len)
        region_text = full_text[region_start:region_end]

        # Prefer the model's own "Step N:" labeling within the think region.
        marker_starts = [(region_start + m.start(), int(m.group(1))) for m in STEP_RE.finditer(region_text)]
        if marker_starts:
            ordinal = 0
            for s, n in marker_starts:
                if s >= region_end:
                    break
                ordinal += 1
                res.steps.append(_make_pos("step", ordinal, n, s, offsets, prompt_len))
            return res

        # Fallback: model didn't use "Step N:" labels -- segment by sentence instead.
        starts = [region_start] + [region_start + m.end() for m in SENT_RE.finditer(region_text)]
        seen, ordinal = set(), 0
        for s in starts:
            if s >= region_end or s in seen or len(full_text[s:region_end].strip()) < 4:
                continue
            seen.add(s)
            ordinal += 1
            res.steps.append(_make_pos("step", ordinal, None, s, offsets, prompt_len))
        return res

    # Answer marker: first `Answer:`; fall back to `</think>` boundary.
    ans_char = full_text.find(answer_mark)
    if ans_char < 0:
        tclose = full_text.find(THINK_CLOSE)
        ans_char = (tclose + len(THINK_CLOSE)) if tclose >= 0 else -1
    if ans_char >= 0:
        res.answer = _make_pos("answer", 0, None, ans_char, offsets, prompt_len)

    region_end = ans_char if ans_char >= 0 else len(full_text)

    if step_mode == "marker":
        ordinal = 0
        for m in STEP_RE.finditer(full_text):
            if m.start() >= region_end:
                break
            ordinal += 1
            res.steps.append(_make_pos("step", ordinal, int(m.group(1)), m.start(), offsets, prompt_len))
    elif step_mode == "paragraph":
        # Paragraph starts within the reasoning region: position 0, then after each blank line.
        starts = [0] + [m.end() for m in PARA_RE.finditer(full_text) if m.end() < region_end]
        # De-dup and keep only non-empty paragraphs.
        seen, ordinal = set(), 0
        for s in starts:
            if s >= region_end or s in seen or not full_text[s:region_end].strip():
                continue
            seen.add(s)
            ordinal += 1
            res.steps.append(_make_pos("step", ordinal, None, s, offsets, prompt_len))
    else:
        raise ValueError(f"step_mode must be 'think', 'marker' or 'paragraph', got {step_mode}")

    return res
