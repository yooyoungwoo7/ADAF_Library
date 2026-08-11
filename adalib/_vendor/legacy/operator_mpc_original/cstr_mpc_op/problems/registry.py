#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/registry.py
============================================================
Single source of truth for problem instances.
"""

from problems.lotka_problem       import LotkaProblem
from problems.bioreactor_problem  import BioreactorProblem
from problems.cstr_problem        import CSTRProblem
from problems.triple_tank_problem import TripleTankProblem


_REGISTRY = {
    "lotka":       LotkaProblem,
    "bioreactor":  BioreactorProblem,
    "cstr":        CSTRProblem,
    "triple_tank": TripleTankProblem,
}


def get_problem(name):
    """Return a fresh problem instance by lowercase name."""
    name = str(name).lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown problem={name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_problems():
    return sorted(_REGISTRY.keys())

from problems.cstr_mpc_problem import CSTRMPCProblem
_REGISTRY["cstr_mpc"] = CSTRMPCProblem

from problems.triple_tank_mpc_problem import TripleTankMPCProblem
_REGISTRY["triple_tank_mpc"] = TripleTankMPCProblem
