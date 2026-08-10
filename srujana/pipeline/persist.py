"""Tiny helper to persist metric dicts as JSON (numpy-safe, NaN -> null).

Used by Stages 5/6 so their results are saved to disk (not stdout-only) and can be dropped
straight into results.md later without re-running.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, np.floating):
        f = float(o)
        return None if f != f else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float):
        return None if o != o else o  # NaN -> null (valid JSON)
    return o


def save_json(obj, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    return str(path)
