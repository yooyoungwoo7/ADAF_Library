"""
adalib/utils/reference.py
Numerical reference trajectory generation via scipy solve_ivp.

Used by result.plot(reference="solve_ivp") and result.inference_plot().
"""
from __future__ import annotations

import warnings
import numpy as np
from typing import Any, List, Optional, Union


def solve_reference_ivp(
    system: Any,
    x0: Any,
    t_eval: Any,
    params: Any = None,
    controls: Any = None,
    method: str = "BDF",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Compute a reference trajectory via scipy solve_ivp.

    Parameters
    ----------
    system
        ODE system with ``rhs(t, x, u=None, p=None)`` (CallableODESystem)
        or ``rhs_np(t, x, params)`` (built-in legacy systems).
    x0
        Initial state, array-like.
    t_eval
        Time points to evaluate, shape ``(T,)``.  The integration span is
        ``[t_eval[0], t_eval[-1]]``.
    params
        ODE parameters ``p`` (passed as keyword ``p=`` to ``system.rhs``).
    controls
        Constant control ``u`` over the entire interval (passed as keyword
        ``u=`` to ``system.rhs``).  Ignored for legacy systems that use
        ``rhs_np(t, x, params)`` directly.
    method
        scipy solve_ivp method (default ``"BDF"`` — suitable for stiff ODEs).
    rtol, atol
        Solver tolerances.

    Returns
    -------
    y : ndarray, shape ``(T, n_state)``
        Trajectory evaluated at ``t_eval``.
    """
    from scipy.integrate import solve_ivp

    t_eval_arr = np.asarray(t_eval, dtype=float)
    x0_arr     = np.asarray(x0,     dtype=float).ravel()
    t_span     = (float(t_eval_arr[0]), float(t_eval_arr[-1]))

    _rhs = _build_rhs(system, params, controls)

    sol = solve_ivp(
        _rhs,
        t_span,
        x0_arr.tolist(),
        method=method,
        t_eval=t_eval_arr,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        warnings.warn(
            f"solve_ivp ({method}) did not converge: {sol.message}",
            stacklevel=2,
        )

    return np.asarray(sol.y, dtype=float).T   # (T, n_state)


def _build_rhs(system: Any, params: Any, controls: Any):
    """Return a callable (t, x) -> dxdt from system + params + controls."""
    if hasattr(system, "rhs"):
        # CallableODESystem or any system with .rhs(t, x, u=None, p=None)
        u = _broadcast_control(controls)
        p = np.asarray(params, dtype=float) if params is not None else None
        def _rhs(t, x):
            return system.rhs(t, x, u=u, p=p)
        return _rhs

    if hasattr(system, "rhs_np"):
        # Legacy built-in systems use rhs_np(t, x, params_array)
        if params is not None:
            theta = np.asarray(params, dtype=float)
        else:
            theta = np.zeros(1, dtype=float)
        def _rhs_np(t, x):
            return system.rhs_np(t, x, theta)
        return _rhs_np

    raise AttributeError(
        f"System {type(system).__name__!r} has neither .rhs nor .rhs_np. "
        "Provide a system with one of these methods."
    )


def _broadcast_control(controls: Any) -> Optional[List]:
    """Normalise scalar / array-like controls to a list (or None)."""
    if controls is None:
        return None
    if isinstance(controls, (int, float)):
        return [float(controls)]
    arr = np.asarray(controls, dtype=float).ravel()
    return arr.tolist()
