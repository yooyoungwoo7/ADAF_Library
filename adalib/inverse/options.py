"""
adalib/inverse/options.py
InverseOptions — training configuration for run_inverse.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class InverseOptions:
    """Training options for the ADA-based inverse solver.

    Loss
    ----
    lambda_physics : float
        Weight for the physics residual loss  (default 1.0).
    lambda_data : float
        Weight for the observation data loss  (default 10.0).

    ADA Basis (ADAF-seq)
    --------------------
    basis : str
        ``"adaf"`` — Adaptive Fourier basis (default, recommended for inverse).
    n_seg : int
        Number of piecewise segments.
    N_p : int
        Number of ADA panels per segment.
    N_m : int
        Number of Fourier modes in the basis expansion.
    Nt_total : int
        Total number of collocation / output grid points.
    gamma : float
        ADA geometry parameter.
    L : float
        ADA geometry parameter.

    Adam optimizer
    --------------
    epochs : int
        Number of Adam outer epochs per segment.
    adam_inner : int
        Gradient steps per epoch (inner loop).
    adam_lr : float
        Adam learning rate.

    L-BFGS polish
    -------------
    use_lbfgs : bool
        If True, run L-BFGS-B refinement after Adam in each segment.

    Training strategy
    -----------------
    training_strategy : str
        ``"joint"`` (default) — every Adam step updates W and θ simultaneously
        via grad(L_total, W + θ).  ``"alternating"`` — block coordinate descent
        (W-only steps then θ-only steps), kept for comparison.
    data_prefit_steps : int
        W-only Adam steps minimizing L_data only, run once at the start of each
        segment on the first pass.  Warm-starts the ADA trajectory before joint
        parameter estimation.  Default 0 (disabled).
    normalize_data_loss : bool
        Scale each observed-state residual by the RMS of its observations in the
        current segment.  Prevents scale mismatch between states.  Default True.
    normalize_physics_loss : bool
        Scale each state's physics residual by a characteristic magnitude.
        Default False.

    Logging
    -------
    verbose : bool
        Print per-segment progress.
    log_every : int
        Print loss every this many Adam steps (0 = only once per segment).
    param_log_every : int
        Record parameter values every this many total Adam steps.

    Precision
    ---------
    dtype : str
        ``"float64"`` (default) or ``"float32"``.

    Auto-save
    ---------
    output_dir : str, optional
        If provided, all outputs are saved here after training:
        ``<system>_trajectory.npz``, ``<system>_loss_history.csv``,
        ``<system>_param_history.csv``, three PNG plots, and
        ``run_metadata.json``.  Directory is created if it doesn't exist.
    """

    # loss weights
    lambda_physics: float = 1.0
    lambda_data:    float = 10.0

    # basis
    basis:    str   = "adaf"
    n_seg:    int   = 20
    N_p:      int   = 5
    N_m:      int   = 100
    Nt_total: int   = 1000
    gamma:    float = 0.8
    L:        float = 1.0

    # Adam
    epochs:     int   = 5
    adam_inner: int   = 200
    adam_lr:    float = 1e-3

    # L-BFGS
    use_lbfgs: bool = True

    # multi-pass sequential
    n_passes: int = 1

    # early-segment warm repeat
    warm_seg_passes: int = 1  # number of times to repeat training for early segments
    n_warm_segs:     int = 3  # how many leading segments to apply warm_seg_passes to

    # training strategy
    training_strategy:      str  = "joint"   # "joint" or "alternating"
    data_prefit_steps:      int  = 0
    normalize_data_loss:    bool = True
    normalize_physics_loss: bool = False

    # logging
    verbose:        bool = True
    log_every:      int  = 0
    param_log_every: int = 1

    # dtype
    dtype: str = "float64"

    # auto-save
    output_dir:  Optional[str]  = None
    true_params: Optional[dict] = None  # {name: true_value} — dashed reference lines in param plot
