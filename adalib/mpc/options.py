"""
adalib/mpc/options.py
MPCOptions dataclass for run_mpc() workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Tuple


@dataclass
class MPCOptions:
    """Options for the run_mpc() full workflow.

    Default values run a full fresh workflow from scratch including
    operator data generation, training, and closed-loop MPC execution.

    Supported systems and dispatch:
      CSTR              → tracking MPC  (minimize (T_R - T_ref)^2)
      TripleTank        → tracking MPC  (minimize ||h - h_ref||^2)
      FedBatchBioreactor → economic MPC (maximize terminal Ps*Vs)

    target examples:
      {"T_R": 136.0}           CSTR tracking
      [None, None, 136.0, None] CSTR tracking (by position)
      {"h1": 0.5, "h2": 0.5}  TripleTank tracking
    """

    mode: str = "tracking"  # "tracking" | "economic"
    basis: str = "lpa"

    # ── MPC setup ────────────────────────────────────────────────────
    n_steps: int = 20
    horizon: Optional[int] = None          # prediction horizon (None = 1-step greedy)
    target: Optional[Union[Dict, List, Tuple]] = None
    control_bounds: Optional[Union[Dict, Tuple]] = None
    state_bounds: Optional[Union[Dict, Tuple]] = None

    # ── Built-in tracking MPC: gradient / sampling extensions ────────
    # gradient: None → legacy 1-step loops (minimize_scalar / FD-SLSQP).
    #   "autodiff" → horizon-H SLSQP with exact dJ/du by automatic
    #                differentiation through the operator (net + basis).
    #   "fd"       → same horizon-H rollout cost, SLSQP finite differences
    #                (controlled comparison against "autodiff").
    # optimizer="CEM" → sampling-based MPC exploiting batched surrogate
    #   inference: cem_samples candidate sequences evaluated per batched
    #   forward pass, cem_iters refinement iterations per MPC step.
    # Supported systems: cstr, triple_tank.
    gradient: Optional[str] = None         # None | "autodiff" | "fd"
    cem_samples: int = 512
    cem_elites: int = 64
    cem_iters: int = 8
    cem_alpha: float = 0.7
    # optimizer="MPPI" — model-predictive path integral (importance-weighted
    # batched sampling): mppi_samples rollouts per iteration, softmax weights
    # exp(-(J-min J)/mppi_lambda), mppi_iters refinement iterations.
    mppi_samples: int = 512
    mppi_iters: int = 4
    mppi_lambda: float = 1.0
    mppi_noise: float = 0.2

    # ── Operator data generation ──────────────────────────────────────
    n_train: int = 200
    n_val: int = 50
    seed: int = 42
    generate_data: bool = True
    reuse_existing_data: bool = False
    force_rebuild_data: bool = False

    # ── Operator training ─────────────────────────────────────────────
    train_operator: bool = True
    reuse_existing_operator: bool = False
    epochs: int = 300
    batch_size: int = 256
    lr: float = 3e-3
    hidden: int = 256
    n_layers: int = 4
    use_lr_schedule: bool = True

    # ── Closed-loop simulation ────────────────────────────────────────
    run_closed_loop: bool = True

    # ── Paths (None → auto under work_dir) ───────────────────────────
    work_dir: Optional[str] = None
    operator_work_dir: Optional[str] = None
    mpc_result_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    data_dir: Optional[str] = None
    log_dir: Optional[str] = None

    # ── Runtime ──────────────────────────────────────────────────────
    verbose: bool = True

    # ── Generic CallableODESystem MPC (ignored for built-in systems) ──
    control_inputs:        Optional[List[str]]   = None
    controlled_variables:  Optional[List[str]]   = None
    parameter_bounds:      Optional[Dict]        = None
    dt:                    Optional[float]       = None
    tracking_weights:      Optional[List[float]] = None
    control_weights:       Optional[List[float]] = None
    terminal_weights:      Optional[List[float]] = None
    optimizer:             str                   = "SLSQP"
    plant_solver:          str                   = "RK45"

    # LPA basis capacity (generic CallableODESystem path only). These were
    # previously undeclared — only read via getattr(opts, ..., default) in
    # _generic_mpc.py, so passing them to the constructor raised
    # TypeError despite the README documenting them as overridable.
    lpa_n_panels:  int = 8
    lpa_max_order: int = 6
    lpa_nt_seg:    int = 20
