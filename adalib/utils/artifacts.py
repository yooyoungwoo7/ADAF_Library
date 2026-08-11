"""
adalib/utils/artifacts.py
Lightweight .npz artifact helpers for ADALib result objects.
"""
from __future__ import annotations

import os
import glob
import numpy as np
from typing import Any, Dict, List, Optional


def save_npz(path: str, **arrays) -> str:
    """Save named arrays to a compressed .npz file.

    Parameters
    ----------
    path : str
        Destination file path (.npz extension recommended).
    **arrays
        Named arrays to save.

    Returns
    -------
    path : str
        The path written to.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a .npz file and return a plain ``dict`` of arrays.

    Parameters
    ----------
    path : str
        Path to a ``.npz`` file.

    Returns
    -------
    dict
        Keys are array names; values are ``np.ndarray``.
    """
    d = np.load(path, allow_pickle=False)
    return dict(d)


def list_run_artifacts(work_dir: str) -> List[str]:
    """Return a sorted list of all files inside *work_dir* (recursive).

    Parameters
    ----------
    work_dir : str
        Root directory to walk.

    Returns
    -------
    list of str
        Absolute paths of every file found, sorted alphabetically.
    """
    if not os.path.isdir(work_dir):
        return []
    found: List[str] = []
    for root, _dirs, files in os.walk(work_dir):
        for fname in sorted(files):
            found.append(os.path.join(root, fname))
    return found


def resolve_result_path(result: Any, filename: str) -> str:
    """Return a path under *result.paths['work_dir']* for the given filename.

    Falls back to the current directory if no work_dir is available.

    Parameters
    ----------
    result
        Any result object that has a ``.paths`` attribute.
    filename : str
        Target filename (e.g. ``"inference.npz"``).

    Returns
    -------
    str
        ``<work_dir>/<filename>``
    """
    paths = getattr(result, "paths", None) or {}
    work_dir = (paths.get("work_dir")
                or paths.get("result_dir")
                or ".")
    return os.path.join(work_dir, filename)
