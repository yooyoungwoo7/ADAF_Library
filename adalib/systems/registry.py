"""
adalib/systems/registry.py
Named registry of built-in ODE systems: get_system() and list_systems().
"""
from __future__ import annotations

from .lotka_volterra import LotkaVolterra, LotkaVolterraUR
from .euler import EulerRigidBody
from .fedbatch_bioreactor import FedBatchBioreactor
from .cstr import CSTR
from .triple_tank import TripleTank
from .pendulum import DampedPendulum

_REGISTRY: dict[str, type] = {
    "lotka_volterra":      LotkaVolterra,
    "lotka_volterra_ur":   LotkaVolterraUR,
    "euler":               EulerRigidBody,
    "fedbatch_bioreactor": FedBatchBioreactor,
    "cstr":                CSTR,
    "triple_tank":         TripleTank,
    "damped_pendulum":     DampedPendulum,
}

_ALIASES: dict[str, str] = {
    "lotka":             "lotka_volterra",
    "euler_rigid_body":  "euler",
    "bioreactor":        "fedbatch_bioreactor",
    "fedbatch":          "fedbatch_bioreactor",
    "tank":              "triple_tank",
    "pendulum":          "damped_pendulum",
}


def get_system(name: str, **kwargs):
    """
    Instantiate a built-in ODE system by name.

    Parameters
    ----------
    name : str
        Canonical name or alias.  Case-insensitive.
        Canonical names: lotka_volterra, euler, fedbatch_bioreactor, cstr, triple_tank
        Aliases: lotka, euler_rigid_body, bioreactor, fedbatch, tank
    **kwargs
        Forwarded to the system constructor.

    Returns
    -------
    ODESystem instance

    Raises
    ------
    ValueError
        If *name* is not recognized.
    """
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _REGISTRY:
        available = sorted(set(list(_REGISTRY) + list(_ALIASES)))
        raise ValueError(
            f"Unknown system '{name}'. "
            f"Available names: {available}. "
            "Use CallableODESystem to define a custom system."
        )
    return _REGISTRY[key](**kwargs)


def list_systems() -> list[str]:
    """Return sorted canonical names of all built-in registered systems."""
    return sorted(_REGISTRY.keys())


def register_system(name: str, system_cls_or_instance, aliases: list[str] | None = None) -> None:
    """Register a custom ODE system under *name*.

    Parameters
    ----------
    name : str
        Canonical lower-case name used by get_system().
    system_cls_or_instance : type or ODESystem instance
        If a class is given, get_system(name) calls ``system_cls_or_instance(**kwargs)``.
        If an already-instantiated object is given, a wrapper factory is registered.
    aliases : list of str, optional
        Additional short names that should resolve to *name*.
    """
    name = name.lower()
    if isinstance(system_cls_or_instance, type):
        _REGISTRY[name] = system_cls_or_instance
    else:
        obj = system_cls_or_instance
        _REGISTRY[name] = lambda **kw: obj

    if aliases:
        for alias in aliases:
            _ALIASES[alias.lower()] = name
