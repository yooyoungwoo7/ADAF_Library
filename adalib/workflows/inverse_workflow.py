"""
adalib/workflows/inverse_workflow.py

run_inverse() — high-level Physics-Informed Inverse Training workflow.

Canonical usage:
    fwd_result  = run_forward(system, x0, t_span, params=[2.0, 0.04, 1.06, 0.02])
    obs         = data_gen(fwd_result, n_points=200, noise_std=0.01, seed=42)
    inv_result  = run_inverse(system, x0, t_span,
                              params={"alpha": InverseParameter(1.5, lower=0.0),
                                      "beta":  0.04,   # fixed
                                      "gamma": InverseParameter(1.2, lower=0.0),
                                      "delta": 0.02},
                              data=obs)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Union

from ..forward.options  import ForwardOptions
from ..inverse.options  import InverseOptions
from ..inverse.result   import InverseResult


def run_inverse(
    system,
    x0,
    t_span,
    params: Dict[str, Any],
    data,
    options: Optional[InverseOptions] = None,
    **kwargs,
) -> InverseResult:
    """Solve a physics-informed inverse problem using ADA-F basis.

    Parameters
    ----------
    system : ODESystem
        Must implement ``rhs_tf()`` that accepts a ``p`` argument.
    x0 : array-like, shape (n_state,)
        Initial condition.
    t_span : (t0, t1)
        Integration interval.
    params : dict[str, float | InverseParameter]
        ODE parameters.  Use plain floats for known (fixed) parameters and
        :class:`InverseParameter` objects for unknowns to be estimated.

        The dict must be ordered so that ``list(params.values())``
        matches the positional order expected by ``system.rhs_tf(..., p=...)``.
        For named systems (LotkaVolterra, EulerRigidBody) the order is
        documented in ``system.parameter_names``.
    data : ObservationData
        Observations produced by :func:`data_gen` or constructed directly.
    options : InverseOptions, optional
        Training configuration.  Any field can also be passed as a kwarg.
    **kwargs
        Override any :class:`InverseOptions` field by name.

    Returns
    -------
    InverseResult
        Contains ``t``, ``y``, ``estimated_params``, ``parameter_history``,
        ``loss_history``, and a ``plot()`` method.

    Examples
    --------
    Lotka-Volterra with unknown alpha and gamma:

    >>> from adalib import run_forward, run_inverse, data_gen
    >>> from adalib import InverseParameter, InverseOptions, get_system
    >>>
    >>> system = get_system("lotka_volterra")
    >>> fwd = run_forward(system, x0=[0.5, 0.075], t_span=(0, 1),
    ...                   params=[2.0, 0.04, 1.06, 0.02])
    >>> obs = data_gen(fwd, n_points=200, noise_std=0.01, seed=42)
    >>>
    >>> inv = run_inverse(
    ...     system, x0=[0.5, 0.075], t_span=(0, 1),
    ...     params={
    ...         "alpha": InverseParameter(1.5, lower=0.0),
    ...         "beta":  0.04,
    ...         "gamma": InverseParameter(1.2, lower=0.0),
    ...         "delta": 0.02,
    ...     },
    ...     data=obs,
    ...     options=InverseOptions(n_seg=10, epochs=3, adam_inner=100),
    ... )
    >>> print(inv.estimated_params)   # {"alpha": ~2.0, "gamma": ~1.06}
    """
    from ..inverse.parameter import InverseParameter
    from ..inverse.solver    import InverseSolver

    # ── Merge options with kwargs ────────────────────────────────────────
    opts = options or InverseOptions()
    for k, v in kwargs.items():
        if hasattr(opts, k):
            setattr(opts, k, v)

    # ── Build parameter lists ────────────────────────────────────────────
    #
    # params_resolved is the flat list passed to rhs_tf as p=...
    # inverse_params  is {name: InverseParameter} for the trainable subset
    #
    x0_np = np.asarray(x0, dtype=float).ravel()

    params_resolved: List[Any] = []
    inverse_params:  Dict[str, InverseParameter] = {}

    DTYPE = _parse_dtype(opts.dtype)
    for name, val in params.items():
        if isinstance(val, InverseParameter):
            val.build(dtype=DTYPE, name=name)
            params_resolved.append(val)
            inverse_params[name] = val
        else:
            params_resolved.append(float(val))

    if len(inverse_params) == 0:
        raise ValueError(
            "run_inverse: params dict contains no InverseParameter objects.  "
            "At least one parameter must be unknown.  "
            "Use run_forward() for fully-known systems."
        )

    # ── Run inverse solver ───────────────────────────────────────────────
    solver = InverseSolver(system, basis=opts.basis)
    raw = solver.solve(
        x0               = list(x0_np),
        t_span           = t_span,
        params_resolved  = params_resolved,
        inverse_params   = inverse_params,
        data             = data,
        n_seg            = opts.n_seg,
        N_p              = opts.N_p,
        N_m              = opts.N_m,
        Nt_total         = opts.Nt_total,
        gamma            = opts.gamma,
        L                = opts.L,
        lambda_physics   = opts.lambda_physics,
        lambda_data      = opts.lambda_data,
        epochs           = opts.epochs,
        adam_inner       = opts.adam_inner,
        adam_lr          = opts.adam_lr,
        use_lbfgs        = opts.use_lbfgs,
        n_passes         = opts.n_passes,
        dtype            = opts.dtype,
        verbose          = opts.verbose,
        param_log_every  = opts.param_log_every,
        training_strategy      = opts.training_strategy,
        data_prefit_steps      = opts.data_prefit_steps,
        normalize_data_loss    = opts.normalize_data_loss,
        normalize_physics_loss = opts.normalize_physics_loss,
        warm_seg_passes        = opts.warm_seg_passes,
        n_warm_segs            = opts.n_warm_segs,
    )

    # ── Wrap result ──────────────────────────────────────────────────────
    metadata = {
        "system_name":  getattr(system, "name", type(system).__name__),
        "state_names":  list(getattr(system, "state_names", [])),
        "t_span":       list(t_span),
        "x0":           list(x0_np),
        "basis":        opts.basis,
        "n_seg":        opts.n_seg,
        "lambda_physics": opts.lambda_physics,
        "lambda_data":    opts.lambda_data,
    }

    result = InverseResult(raw, inverse_params, metadata=metadata)

    if opts.output_dir:
        result.save_all(
            output_dir=opts.output_dir,
            observation_data=data,
            true_params=opts.true_params,
        )
        if opts.verbose:
            print(f"[run_inverse] outputs saved → {opts.output_dir}")

    return result


# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str):
    import tensorflow as tf
    d = dtype_str.lower()
    if d in ("float64", "fp64", "double"):
        return tf.float64
    return tf.float32
