"""
adalib/workflows/dispatch.py
Unified run() dispatcher that delegates to run_forward / run_operator / run_mpc.
"""
from __future__ import annotations

from .forward_workflow  import run_forward
from .operator_workflow import run_operator
from .mpc_workflow      import run_mpc
from .inverse_workflow  import run_inverse


def run(
    feature: str,
    system,
    x0=None,
    t_span=None,
    params=None,
    controls=None,
    options=None,
    **kwargs,
):
    """Unified dispatcher for all three ADALib features.

    Parameters
    ----------
    feature : {"forward", "operator", "mpc"}
        Which workflow to run.
    system : str or ODESystem
        System to solve / learn / control.
    x0 : array-like
        Initial state.
    t_span : (t0, t1)
        Integration or observation time span.
    params : array-like, optional
        ODE parameters.  Also accepted as kwarg ``p``.
    controls : array-like, optional
        External control sequence.  Also accepted as kwarg ``u``.
    options : dataclass or dict, optional
        Feature-specific options.  Can also pass any option field as a kwarg.
    **kwargs
        Override any options field by name.

    Returns
    -------
    ForwardResult | OperatorResult | MPCResult
        The result object for the selected feature.

    Examples
    --------
    >>> result = run("forward", "lotka_volterra", x0=[0.5, 0.075],
    ...              t_span=(0, 1), params=[2, 0.04, 1.06, 0.02])

    >>> result = run("operator", "cstr",
    ...              x0=[0.8, 0.5, 134.14, 130.0], t_span=(0, 0.5),
    ...              params=[1.0, 1.0, 50.0, -2000.0],
    ...              options=OperatorOptions(n_train=100, epochs=10))

    >>> result = run("mpc", "cstr",
    ...              x0=[0.8, 0.5, 141.0, 141.0],
    ...              options=MPCOptions(n_steps=5))
    """
    feature = feature.lower().strip()

    if feature == "forward":
        return run_forward(
            system, x0=x0, t_span=t_span,
            params=params or kwargs.pop("p", None),
            options=options, **kwargs,
        )
    elif feature in ("operator", "op"):
        return run_operator(
            system, x0=x0, t_span=t_span,
            params=params or kwargs.pop("p", None),
            controls=controls or kwargs.pop("u", None),
            options=options, **kwargs,
        )
    elif feature in ("mpc", "control"):
        return run_mpc(
            system, x0=x0, t_span=t_span,
            options=options, **kwargs,
        )
    elif feature in ("inverse", "inv"):
        return run_inverse(
            system, x0=x0, t_span=t_span,
            params=params or kwargs.pop("p", {}),
            data=kwargs.pop("data", None),
            options=options, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown feature={feature!r}.  "
            "Choose one of: 'forward', 'operator', 'mpc', 'inverse'."
        )
