from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Solution:
    """
    solve_ivp-like result container.

    t : (Nt,)
    y : (n_state, Nt)  (SciPy solve_ivp와 동일한 shape)
    """
    t: np.ndarray
    y: np.ndarray
    status: int = 0
    message: str = "success"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_state(self) -> int:
        return int(self.y.shape[0])

    @property
    def Nt(self) -> int:
        return int(self.y.shape[1])

    def as_timeseries(self) -> np.ndarray:
        """
        Return y as (Nt, n_state) (기존 너희 내부 y_pred 형태로 보고 싶을 때)
        """
        return self.y.T