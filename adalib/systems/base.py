"""
adalib/systems/base.py
Common ODE system interface for forward solving, operator learning, and MPC.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp


class ODESystem(ABC):
    name: str = ""
    state_names: List[str] = []
    control_names: List[str] = []
    parameter_names: List[str] = []

    state_bounds: dict = {}
    control_bounds: dict = {}
    parameter_bounds: dict = {}

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @abstractmethod
    def rhs(self, t: float, x: np.ndarray, u=None, p=None) -> np.ndarray:
        raise NotImplementedError

    def rhs_tf(self, var_list, i: int, u=None, p=None):
        """
        TF-native residual for state i (XLA-compatible, supports L-BFGS).
        var_list[k] = (y_k, y_k_t) — TF tensors of shape (Nt_seg,)
        Returns: dy_i/dt - f_i(y) as a TF tensor.
        Subclasses must override this to enable ForwardSolver.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement rhs_tf() for ForwardSolver. "
            "See adalib/systems/lotka_volterra.py for an example."
        )

    def simulate(
        self,
        x0: np.ndarray,
        t_span,
        u=None,
        p=None,
        t_eval=None,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        def _rhs(t, x):
            return self.rhs(t, x, u=u, p=p)
        return scipy_solve_ivp(_rhs, t_span, x0, t_eval=t_eval,
                                method=method, rtol=rtol, atol=atol)
