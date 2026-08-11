"""
adalib/systems/callable_system.py
ODE system defined by user-supplied Python callables.
"""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from .base import ODESystem


class CallableODESystem(ODESystem):
    """
    ODE system defined by a Python callable.

    Parameters
    ----------
    name : str
        Human-readable name for this system.
    rhs : callable (t, x, u=None, p=None) -> array-like
        NumPy/Python function returning dx/dt.
        ``p`` can be a list, dict, or any Python object.
    state_names : list[str]
        Names for each state variable.
    control_names : list[str], optional
    parameter_names : list[str], optional
    state_bounds : dict, optional
    control_bounds : dict, optional
    parameter_bounds : dict, optional
    rhs_tf : callable (var_list, i, u=None, p=None) -> TF tensor, optional
        TensorFlow-native residual for XLA-compatible L-BFGS training.
        ``var_list[k] = (y_k, y_k_t)`` — shape (Nt_seg,) float tensors.
        Must return ``dy_i/dt - f_i(y)`` as a TF tensor.

        If omitted, an **automatic numpy bridge** is used via
        ``tf.numpy_function`` so you do NOT need to write this.
        The trade-off: L-BFGS is automatically disabled (XLA-incompatible),
        and Adam-only training is used instead.  For most problems this is
        sufficient.  Provide ``rhs_tf`` only when you need maximum accuracy.

    has_native_rhs_tf : bool (read-only property)
        ``True`` when ``rhs_tf`` was explicitly provided.
        ``run_forward`` checks this to decide whether L-BFGS is safe.

    Note
    ----
    The auto numpy bridge assumes an **autonomous ODE** (RHS does not
    depend on ``t``).  If your system is non-autonomous, provide ``rhs_tf``
    explicitly.

    Example — minimal (no rhs_tf needed)
    --------------------------------------
    >>> from adalib.systems import CallableODESystem
    >>> from adalib.workflows import run_forward
    >>>
    >>> def my_rhs(t, x, u=None, p=None):
    ...     a, b = p["a"], p["b"]
    ...     return [a*x[0] - b*x[0]*x[1], -x[1] + x[0]*x[1]]
    >>>
    >>> system = CallableODESystem("lv", my_rhs, state_names=["x1", "x2"],
    ...                            parameter_names=["a", "b"])
    >>> result = run_forward(system, x0=[1., 0.5], t_span=(0., 10.),
    ...                      params={"a": 1., "b": 0.2})
    """

    def __init__(
        self,
        name: str,
        rhs: Callable,
        state_names: List[str],
        control_names: Optional[List[str]] = None,
        parameter_names: Optional[List[str]] = None,
        state_bounds: Optional[dict] = None,
        control_bounds: Optional[dict] = None,
        parameter_bounds: Optional[dict] = None,
        rhs_tf: Optional[Callable] = None,
    ):
        self.name = name
        self._rhs_fn = rhs
        self._rhs_tf_fn = rhs_tf
        self.state_names = list(state_names)
        self.control_names = list(control_names) if control_names else []
        self.parameter_names = list(parameter_names) if parameter_names else []
        self.state_bounds = state_bounds or {}
        self.control_bounds = control_bounds or {}
        self.parameter_bounds = parameter_bounds or {}

    # ------------------------------------------------------------------
    @property
    def has_native_rhs_tf(self) -> bool:
        """True when a TF-native rhs_tf was explicitly provided."""
        return self._rhs_tf_fn is not None

    # ------------------------------------------------------------------
    def rhs(self, t, x, u=None, p=None) -> np.ndarray:
        result = self._rhs_fn(t, x, u=u, p=p)
        return np.asarray(result, dtype=float)

    def rhs_tf(self, var_list, i: int, u=None, p=None):
        if self._rhs_tf_fn is not None:
            return self._rhs_tf_fn(var_list, i, u=u, p=p)

        # ── Automatic numpy bridge ────────────────────────────────────
        # Wraps self._rhs_fn in tf.numpy_function so the solver can call
        # it inside a TF graph.  XLA-incompatible → L-BFGS is disabled
        # by run_forward automatically.
        import tensorflow as tf

        n_states = len(var_list)
        y_i, y_i_t = var_list[i]
        dtype = y_i.dtype

        # Stack all states column-wise → (Nt_seg, n_states)
        y_all = tf.stack([var_list[k][0] for k in range(n_states)], axis=1)

        rhs_fn = self._rhs_fn  # captured in closure

        def _eval_fi(y_np):
            # y_np: (Nt_seg, n_states) numpy array
            fi = np.array(
                [rhs_fn(0.0, y_np[j].tolist(), u=u, p=p)[i]
                 for j in range(len(y_np))],
                dtype=y_np.dtype,
            )
            return fi

        fi = tf.numpy_function(_eval_fi, [y_all], dtype)
        fi.set_shape(y_i.shape)
        return y_i_t - fi
