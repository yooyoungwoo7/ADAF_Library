"""
adalib/workflows/forward_workflow.py
High-level run_forward() convenience function.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, Optional

from ..forward.solver import ForwardSolver
from ..forward.options import ForwardOptions


# ── Result wrapper ──────────────────────────────────────────────────────

class ForwardResult:
    """Thin wrapper around the raw ADA-F/LPA solver result.

    Attributes
    ----------
    solution
        The raw pinn_lib solver result.  ``solution.t`` and ``solution.y``
        are the canonical time and state arrays.
    t : ndarray
        Alias for ``solution.t``.
    y : ndarray
        Alias for ``solution.y``.
    metadata : dict
        System name, state names, t_span, x0, params.
    paths : dict
        Work-dir paths if any.

    Methods
    -------
    plot(reference, state_names, save_path, show, title, **kwargs)
        Plot state trajectories, optionally vs a reference.
    to_arrays()
        Return ``(t, y)`` as plain numpy arrays.
    save_npz(path)
        Save ``t`` and ``y`` to a compressed .npz file.
    list_artifacts()
        List all files under ``paths["work_dir"]``.
    """

    def __init__(
        self,
        solution: Any,
        metadata: Optional[Dict] = None,
        paths: Optional[Dict]    = None,
        _system: Any             = None,
        _x0: Any                 = None,
        _params: Any             = None,
    ):
        self.solution  = solution
        self.metadata  = metadata or {}
        self.paths     = paths    or {}
        self._system   = _system
        self._x0       = _x0
        self._params   = _params

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def t(self) -> np.ndarray:
        return self.solution.t

    @property
    def y(self) -> np.ndarray:
        return self.solution.y

    # ── Post-processing methods ───────────────────────────────────────

    def plot(
        self,
        reference: Any       = None,
        state_names          = None,
        save_path            = None,
        show: bool           = False,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Plot state trajectories, optionally vs a reference.

        Parameters
        ----------
        reference
            * ``"solve_ivp"`` — compute reference using the stored system
              and ``x0`` via scipy BDF (requires system+x0 stored at run-time).
            * ``callable(t) -> array`` — evaluated on ``result.t``.
            * ``(t_ref, y_ref)`` tuple or scipy OdeResult.
            * ``None`` — trajectory only.
        state_names
            Override axis labels.
        save_path
            If given, save figure here.
        show
            If True, call ``plt.show()``.
        title
            Figure suptitle.

        Returns
        -------
        (fig, axes)
        """
        from ..utils.plotting import plot_forward_result
        _ref = self._resolve_reference(reference)
        return plot_forward_result(
            self,
            reference   = _ref,
            state_names = state_names,
            save_path   = save_path,
            show        = show,
            title       = title,
            **kwargs,
        )

    def forward_plot(
        self,
        state_names          = None,
        save_path            = None,
        show: bool           = False,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Plot ADAF result vs scipy reference — no boilerplate needed.

        Automatically computes a scipy BDF reference from the stored system
        and initial condition.  All arguments are optional.

        Returns
        -------
        (fig, axes)
        """
        return self.plot(
            reference   = "solve_ivp",
            state_names = state_names,
            save_path   = save_path,
            show        = show,
            title       = title,
            **kwargs,
        )

    def to_arrays(self):
        """Return ``(t, y)`` as numpy arrays.

        Returns
        -------
        t : ndarray, shape ``(Nt,)``
        y : ndarray, shape ``(n_state, Nt)``
        """
        return np.asarray(self.t), np.asarray(self.y)

    def save_npz(self, path: Optional[str] = None) -> str:
        """Save ``t`` and ``y`` to a compressed .npz file.

        Parameters
        ----------
        path
            Destination.  Defaults to
            ``<work_dir>/forward_result.npz``.

        Returns
        -------
        str
            Path written to.
        """
        from ..utils.artifacts import save_npz
        if path is None:
            work = self.paths.get("work_dir", ".")
            path = os.path.join(work, "forward_result.npz")
        save_npz(path, t=self.t, y=self.y)
        return path

    def list_artifacts(self):
        """List all files under ``paths['work_dir']``."""
        from ..utils.artifacts import list_run_artifacts
        work = self.paths.get("work_dir")
        return list_run_artifacts(work) if work else []

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_reference(self, reference):
        """Pre-compute reference trajectory when reference='solve_ivp'."""
        if reference != "solve_ivp":
            return reference

        import warnings
        if self._system is None or self._x0 is None:
            warnings.warn(
                "reference='solve_ivp' requires system and x0 to be stored in "
                "the result.  They are stored automatically when run_forward() "
                "is called normally.",
                stacklevel=3,
            )
            return None

        from ..utils.reference import solve_reference_ivp
        t_arr = np.asarray(self.t, dtype=float)
        try:
            y_ref = solve_reference_ivp(
                self._system, self._x0, t_arr,
                params=self._params, method="BDF",
            )
            return (t_arr, y_ref)
        except Exception as exc:
            warnings.warn(
                f"solve_reference_ivp failed: {exc}",
                stacklevel=3,
            )
            return None


# ── Public API ───────────────────────────────────────────────────────────

def run_forward(
    system,
    x0,
    t_span,
    params=None,
    options: ForwardOptions | None = None,
    **kwargs,
) -> ForwardResult:
    """Solve an ODE forward in time using the ADA-F or LPA basis.

    Parameters
    ----------
    system : ODESystem
        Must implement ``rhs_tf()`` for L-BFGS / ADA training.
    x0 : array-like
        Initial state vector.
    t_span : (t0, t1)
        Integration interval.
    params : list, optional
        ODE parameters *p* passed to ``system.rhs`` / ``system.rhs_tf``.
    options : ForwardOptions, optional
        Training options.  Keyword arguments override individual fields.
    **kwargs
        Override any :class:`ForwardOptions` field by name.

    Returns
    -------
    ForwardResult
        Wraps the pinn_lib solver result.  ``result.solution.t`` and
        ``result.solution.y`` are ``ndarray`` of shape ``(Nt_total,)``
        and ``(n_state, Nt_total)`` respectively.  Convenience aliases
        ``result.t`` and ``result.y`` are also available.

        Call ``result.plot(reference="solve_ivp")`` to compare against a
        numerical reference.

    Example
    -------
    >>> from adalib import run_forward, get_system
    >>> system = get_system("euler", I1=0.2, I2=0.3, I3=0.4)
    >>> result = run_forward(system, x0=[1, 1, 1], t_span=(0, 2.5))
    >>> t, y = result.t, result.y
    >>> result.plot(save_path="forward.png")
    """
    opts = options or ForwardOptions()
    for k, v in kwargs.items():
        if hasattr(opts, k):
            setattr(opts, k, v)

    # When the system has no TF-native rhs_tf, fall back to numpy bridge.
    # tf.numpy_function is XLA-incompatible → disable both XLA and L-BFGS.
    uses_native_tf = getattr(system, "has_native_rhs_tf", True)
    xla_predict = True
    if not uses_native_tf:
        xla_predict = False
        if opts.use_lbfgs:
            import warnings
            warnings.warn(
                f"System '{getattr(system, 'name', type(system).__name__)}' has no "
                "native rhs_tf — using auto numpy bridge (XLA disabled). "
                "L-BFGS also disabled. Provide rhs_tf=... to enable both.",
                stacklevel=2,
            )
            opts.use_lbfgs = False

    solver = ForwardSolver(system, basis=opts.basis)
    raw = solver.solve(
        x0=x0,
        t_span=t_span,
        p=params,
        n_seg=opts.n_seg,
        N_p=opts.N_p,
        N_m=opts.N_m,
        Nt_total=opts.Nt_total,
        epochs=opts.epochs,
        adam_inner=opts.adam_inner,
        adam_lr=opts.adam_lr,
        use_lbfgs=opts.use_lbfgs,
        dtype=opts.dtype,
        verbose=opts.verbose,
        gamma=opts.gamma,
        L=opts.L,
        order=opts.order,
        xla_predict=xla_predict,
    )

    metadata = {
        "system_name": getattr(system, "name", type(system).__name__),
        "state_names": list(getattr(system, "state_names", [])),
        "t_span":      list(t_span),
        "x0":          list(np.asarray(x0).ravel()),
        "params":      (list(np.asarray(params).ravel())
                        if params is not None else None),
        "basis":       opts.basis,
        "n_seg":       opts.n_seg,
        "epochs":      opts.epochs,
    }

    return ForwardResult(
        solution = raw.solution,   # raw is Solver; .solution has .t and .y
        metadata = metadata,
        paths    = {},
        _system  = system,
        _x0      = x0,
        _params  = params,
    )
