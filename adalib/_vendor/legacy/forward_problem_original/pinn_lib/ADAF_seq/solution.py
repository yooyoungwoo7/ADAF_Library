from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Solution:
    """
    solve_ivp-like result container.

    t : (Nt,)
    y : (n_state, Nt)
    solver : backend solver object
    """
    t: np.ndarray
    y: np.ndarray
    status: int = 0
    message: str = "success"
    meta: Dict[str, Any] = field(default_factory=dict)
    solver: Optional[Any] = None

    @property
    def n_state(self) -> int:
        return int(self.y.shape[0])

    @property
    def Nt(self) -> int:
        return int(self.y.shape[1])

    def as_timeseries(self) -> np.ndarray:
        """
        Return y as (Nt, n_state)
        """
        return self.y.T

    def __getattr__(self, name):
        """
        If attribute is not found in Solution itself,
        try to access it from backend solver.
        """
        if self.solver is not None and hasattr(self.solver, name):
            return getattr(self.solver, name)
        raise AttributeError(f"'Solution' object has no attribute '{name}'")