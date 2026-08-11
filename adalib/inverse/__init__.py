"""
adalib/inverse — Physics-Informed Inverse Training

Public API:
    ObservationData   — standardized observation container
    InverseParameter  — user-facing trainable ODE parameter wrapper
    InverseOptions    — training options for run_inverse
    InverseResult     — result object returned by run_inverse
    data_gen          — extract observations from a ForwardResult
"""

from .observation import ObservationData, data_gen
from .parameter import InverseParameter
from .options import InverseOptions
from .result import InverseResult

__all__ = [
    "ObservationData",
    "InverseParameter",
    "InverseOptions",
    "InverseResult",
    "data_gen",
]
