"""
adalib/operator/model.py
Re-exports OperatorLearner from the legacy operator package.
"""
from __future__ import annotations
import sys

from ..utils.paths import legacy_operator_root as _legacy_operator_root

_OP_PATH = str(_legacy_operator_root())
if _OP_PATH not in sys.path:
    sys.path.insert(0, _OP_PATH)

from models.learner import OperatorLearner
from models.operator_net import OperatorNet

__all__ = ["OperatorLearner", "OperatorNet"]
