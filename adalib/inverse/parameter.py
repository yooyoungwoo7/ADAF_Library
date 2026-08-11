"""
adalib/inverse/parameter.py

InverseParameter — user-facing wrapper for a trainable ODE parameter.

Users never create tf.Variable directly; they declare:

    params = {
        "alpha": InverseParameter(initial=1.5, lower=0.0),
        "beta":  InverseParameter(initial=0.05, lower=0.0, upper=1.0),
        "gamma": 1.06,   # fixed: plain Python scalar
    }

Fixed values (plain scalars) and InverseParameter objects can coexist
in the same dict.  run_inverse() handles the dispatch.

Constraint transforms
---------------------
- No bounds          → raw tf.Variable (identity)
- lower only         → lower + softplus(raw)
- upper only         → upper - softplus(-raw)
- lower AND upper    → lower + (upper - lower) * sigmoid(raw)

This ensures the constrained value stays in [lower, upper] while
gradients flow through the transform into the optimizer.

The spec requirement: inverse parameter must never be converted to
Python float / numpy before entering ODE RHS.  Use .constrained
(a TF tensor) wherever ODE evaluation occurs.
"""
from __future__ import annotations

import math
from typing import Optional


class InverseParameter:
    """Trainable ODE parameter for inverse problems.

    Parameters
    ----------
    initial : float
        Starting value of the parameter (in physical / constrained space).
    lower : float, optional
        Lower bound.  The parameter is guaranteed >= lower during training.
    upper : float, optional
        Upper bound.  The parameter is guaranteed <= upper during training.
    name : str, optional
        Label used in result reporting and parameter history.

    Examples
    --------
    >>> p = InverseParameter(initial=1.5, lower=0.0)
    >>> p.build(dtype=tf.float64)
    >>> print(p.numpy_value)   # 1.5
    """

    def __init__(
        self,
        initial: float,
        lower:   Optional[float] = None,
        upper:   Optional[float] = None,
        name:    Optional[str]   = None,
    ):
        self.initial = float(initial)
        self.lower   = float(lower) if lower is not None else None
        self.upper   = float(upper) if upper is not None else None
        self.name    = name
        self._raw_var = None  # created in build()

    # ------------------------------------------------------------------

    def build(self, dtype=None, name: Optional[str] = None):
        """Create the underlying tf.Variable.

        Must be called once before training.  Subsequent calls are no-ops.

        Parameters
        ----------
        dtype : tf.DType, optional
            Defaults to tf.float64.
        name : str, optional
            Override the variable name.

        Returns
        -------
        tf.Variable  (the raw / unconstrained variable)
        """
        if self._raw_var is not None:
            return self._raw_var

        import tensorflow as tf

        if dtype is None:
            dtype = tf.float64

        raw_init = self._compute_raw_initial()
        var_name = name or self.name or "inv_param"
        self._raw_var = tf.Variable(
            raw_init, dtype=dtype, trainable=True, name=var_name
        )
        return self._raw_var

    def reset(self):
        """Reset the variable to its initial value (useful for re-runs)."""
        if self._raw_var is not None:
            import tensorflow as tf
            raw_init = self._compute_raw_initial()
            self._raw_var.assign(tf.cast(raw_init, self._raw_var.dtype))

    # ------------------------------------------------------------------

    def _compute_raw_initial(self) -> float:
        """Compute the raw (unconstrained) initial value for the transform."""
        v = self.initial
        lo = self.lower
        hi = self.upper

        if lo is not None and hi is not None:
            # sigmoid: constrained = lo + (hi-lo)*sigmoid(raw)
            # sigmoid(raw) = (v - lo) / (hi - lo)
            t = (v - lo) / (hi - lo)
            t = max(1e-6, min(1 - 1e-6, t))
            return math.log(t / (1.0 - t))  # logit

        if lo is not None:
            # softplus: constrained = lo + softplus(raw)
            # softplus(raw) = v - lo → raw = softplus_inv(v - lo)
            t = max(v - lo, 1e-6)
            # softplus_inverse(t) = log(exp(t) - 1)  ≈ t for large t
            return math.log(math.expm1(max(t, 1e-6)) + 1e-30)

        if hi is not None:
            # softplus: constrained = hi - softplus(-raw)
            # softplus(-raw) = hi - v → -raw = softplus_inv(hi-v)
            t = max(hi - v, 1e-6)
            return -math.log(math.expm1(max(t, 1e-6)) + 1e-30)

        return v  # unconstrained

    # ------------------------------------------------------------------

    @property
    def constrained(self):
        """Current constrained value as a TF tensor (gradient-safe).

        Use this everywhere ODE RHS is evaluated; never call .numpy().
        """
        if self._raw_var is None:
            raise RuntimeError(
                "InverseParameter has not been built yet.  "
                "Call .build() or use run_inverse() which builds automatically."
            )
        import tensorflow as tf

        raw = self._raw_var
        dtype = raw.dtype

        if self.lower is not None and self.upper is not None:
            lo = tf.constant(self.lower, dtype=dtype)
            hi = tf.constant(self.upper, dtype=dtype)
            return lo + (hi - lo) * tf.sigmoid(raw)

        if self.lower is not None:
            lo = tf.constant(self.lower, dtype=dtype)
            return lo + tf.nn.softplus(raw)

        if self.upper is not None:
            hi = tf.constant(self.upper, dtype=dtype)
            return hi - tf.nn.softplus(-raw)

        return raw  # identity

    @property
    def numpy_value(self) -> float:
        """Current value as a Python float (read-only snapshot)."""
        return float(self.constrained.numpy())

    # ------------------------------------------------------------------

    def __repr__(self):
        bounds = ""
        if self.lower is not None:
            bounds += f", lower={self.lower}"
        if self.upper is not None:
            bounds += f", upper={self.upper}"
        built = "built" if self._raw_var is not None else "not built"
        return (
            f"InverseParameter(initial={self.initial}{bounds}, "
            f"name={self.name!r}, {built})"
        )
