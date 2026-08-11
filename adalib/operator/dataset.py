"""
adalib/operator/dataset.py
Re-exports dataset builder from legacy package.
"""
from __future__ import annotations
import sys

from ..utils.paths import legacy_operator_root as _legacy_operator_root

_OP_PATH = str(_legacy_operator_root())
if _OP_PATH not in sys.path:
    sys.path.insert(0, _OP_PATH)

from data.dataset_builder import (
    load_segments,
    load_fullcase,
    build_and_save_fullcase,
    build_segments_from_fullcase,
    make_tf_dataset,
)

__all__ = [
    "load_segments",
    "load_fullcase",
    "build_and_save_fullcase",
    "build_segments_from_fullcase",
    "make_tf_dataset",
]
