"""
adalib/utils/metrics.py — re-exports from legacy operator package.
"""
import sys
from .paths import legacy_operator_root as _legacy_operator_root

_OP = str(_legacy_operator_root())
if _OP not in sys.path:
    sys.path.insert(0, _OP)

from utils.metrics import l2_rel, statewise_l2_rel
__all__ = ["l2_rel", "statewise_l2_rel"]
