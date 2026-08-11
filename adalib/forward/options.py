"""
adalib/forward/options.py
Dataclass options for ForwardSolver.solve().
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ForwardOptions:
    """
    Options for :meth:`ForwardSolver.solve`.

    Attributes
    ----------
    basis : str
        ``'adaf'`` — Adaptive Fourier, piecewise sequential.
        ``'lpa'``  — Legendre Polynomial, full-domain.
    n_seg : int
        Number of piecewise segments (ADAF only).
    N_p : int
        Fourier/Legendre mode count.
    N_m : int
        Fourier mode truncation (ADAF only).
    Nt_total : int
        Total time-grid points across the domain.
    epochs : int
        Adam outer epochs.
    adam_inner : int
        Adam gradient steps per epoch.
    adam_lr : float
        Adam learning rate.
    use_lbfgs : bool
        L-BFGS polish after Adam.  Requires ``rhs_tf`` (default True).
    dtype : str
        ``'float32'`` or ``'float64'``.
    verbose : bool
        Print per-epoch training progress.
    gamma : float
        ADAF / LPA time-scale parameter.
    L : float
        ADAF shift parameter.
    order : int
        LPA polynomial order.
    """
    basis: str      = "adaf"
    n_seg: int      = 50
    N_p: int        = 5
    N_m: int        = 100
    Nt_total: int   = 2500
    epochs: int     = 5
    adam_inner: int = 100
    adam_lr: float  = 1e-3
    use_lbfgs: bool = True
    dtype: str      = "float64"
    verbose: bool   = True
    gamma: float    = 0.8
    L: float        = 1.0
    order: int      = 3
