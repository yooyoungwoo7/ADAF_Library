"""
SystemSpec (validated user input) -> parsed equations -> adalib system object.

``parse_spec`` is import-light and is used by the API for validation.
``build_callable_system`` imports adalib and is only called inside workers.
"""
from __future__ import annotations

from typing import List, Tuple

from ..schemas.common import EquationIssue, SystemSpec, ValidateResponse
from .equation_parser import EquationError, ParsedSystem, parse_system


def parse_spec(spec: SystemSpec) -> ParsedSystem:
    return parse_system(spec.state_names, spec.parameter_names, spec.equations, spec.control_names)


def validate_spec(spec: SystemSpec) -> ValidateResponse:
    try:
        parsed = parse_spec(spec)
    except EquationError as exc:
        return ValidateResponse(ok=False, errors=[EquationIssue(message=str(exc), index=exc.index,
                                                                field=exc.field_name)])
    used = set().union(*(eq.symbols for eq in parsed.equations))
    return ValidateResponse(
        ok=True,
        latex=[eq.latex for eq in parsed.equations],
        symbols=[sorted(eq.symbols) for eq in parsed.equations],
        unused_parameters=[p for p in parsed.parameter_names + parsed.control_names if p not in used],
    )


def build_callable_system(spec: SystemSpec) -> Tuple[object, List[float], List[float]]:
    """Return ``(CallableODESystem, x0, params)`` with ``params`` ordered like
    ``spec.parameters``. A TF-native ``rhs_tf`` is always supplied, so adalib
    keeps XLA prediction and L-BFGS enabled."""
    from adalib import CallableODESystem

    parsed = parse_spec(spec)
    system = CallableODESystem(
        name=spec.name or "custom_system",
        rhs=parsed.rhs_numpy(),
        state_names=parsed.state_names,
        parameter_names=parsed.parameter_names,
        rhs_tf=parsed.rhs_tensorflow(),
    )
    x0 = [float(s.initial) for s in spec.states]
    params = [float(p.value) for p in spec.parameters]
    return system, x0, params


def build_operator_system(spec: SystemSpec):
    """``CallableODESystem`` for adalib's generic operator / MPC paths.

    Those paths call ``rhs(t, x, u=...)`` *without* ``p`` (adalib 0.1.0,
    mpc/_generic_mpc.py), so parameter values are bound into the closure.
    ``state_bounds``/``control_bounds`` define the training distribution."""
    from adalib import CallableODESystem

    parsed = parse_spec(spec)
    base = parsed.rhs_numpy()
    fixed = [float(p.value) for p in spec.parameters]

    def rhs(t, x, u=None, p=None):
        return base(t, x, u=u, p=fixed if p is None else p)

    state_bounds = {s.name.strip(): (float(s.lower), float(s.upper))
                    for s in spec.states if s.lower is not None and s.upper is not None}
    control_bounds = {c.name.strip(): (float(c.lower), float(c.upper)) for c in spec.controls}
    return CallableODESystem(
        name=spec.name or "custom_system",
        rhs=rhs,
        state_names=parsed.state_names,
        control_names=parsed.control_names,
        parameter_names=[],
        state_bounds=state_bounds,
        control_bounds=control_bounds,
    )
