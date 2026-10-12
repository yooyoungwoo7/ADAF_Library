"""Conversion of NumPy results to JSON-safe structures and export files."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def finite_list(arr) -> List[Optional[float]]:
    """1-D array -> list of floats; NaN/inf become None (JSON has no NaN)."""
    a = np.asarray(arr, dtype=float).ravel()
    return [float(v) if math.isfinite(v) else None for v in a.tolist()]


def downsample_index(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return idx


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_list(obj) if obj.ndim <= 1 else [to_jsonable(r) for r in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return repr(obj)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, columns: Dict[str, Any]) -> None:
    names = list(columns)
    arrays = [np.asarray(columns[n], dtype=float).ravel() for n in names]
    n = min(a.shape[0] for a in arrays)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow([repr(float(a[i])) for a in arrays])
