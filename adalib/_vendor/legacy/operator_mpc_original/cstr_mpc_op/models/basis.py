#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/basis.py
============================================================
Basis factory. Pick a basis at construction via `BASIS_NAME` (or env var
`BASIS=<name>`):

    "lpa"   — W-panel Legendre Polynomial Approximation (default).
              Hard IC: x(0)=x0.  See models/lpa_basis.py.
    "adaf"  — Fourier basis with hard IC + ḋC: x(0)=x0 AND ẋ(0)=ẋ0.
              See models/adaf_basis.py.

Both bases share the call signature `basis(W, x0, xdot0)` and return the
same dict keys, so the learner uses them interchangeably.
"""

from config import BASIS_NAME
from models.lpa_basis import BatchLPABasis
from models.adaf_basis import BatchADAFBasis


_REGISTRY = {
    "lpa":  BatchLPABasis,
    "adaf": BatchADAFBasis,
}


def get_basis_cls(name=None):
    key = (name or BASIS_NAME).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown BASIS_NAME={key!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def build_basis(name=None, **kwargs):
    """Instantiate a basis by name. kwargs forwarded to its constructor."""
    return get_basis_cls(name)(**kwargs)
