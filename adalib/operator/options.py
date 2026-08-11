"""
adalib/operator/options.py
OperatorOptions dataclass for run_operator() workflow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OperatorOptions:
    """Options for the run_operator() full workflow.

    Default values run a full fresh workflow from scratch:
      generate_data=True, train=True, infer=True,
      reuse_existing_data=False, reuse_existing_checkpoint=False.

    Reuse is opt-in and raises clear errors when required files are missing.
    """

    basis: str = "lpa"

    # ── Data generation ──────────────────────────────────────────────
    n_train: int = 2000
    n_val: int = 200
    seed: int = 42
    generate_data: bool = True
    reuse_existing_data: bool = False
    force_rebuild_data: bool = False

    # ── Training ─────────────────────────────────────────────────────
    train: bool = True
    reuse_existing_checkpoint: bool = False
    epochs: int = 1000
    batch_size: int = 256
    lr: float = 3e-3
    hidden: int = 128
    n_layers: int = 3
    use_lr_schedule: bool = True

    # ── Inference ────────────────────────────────────────────────────
    infer: bool = True

    # ── Paths (None → auto under work_dir) ───────────────────────────
    work_dir: Optional[str] = None
    data_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    result_dir: Optional[str] = None

    # ── Runtime ──────────────────────────────────────────────────────
    verbose: bool = True

    # ── Generic CallableODESystem (ignored for built-in systems) ─────
    dt:    Optional[float] = None   # segment duration [s or problem time unit]
    n_seg: int             = 25     # rollout length in segments

    # LPA basis capacity (generic CallableODESystem path only). Defaults
    # (8, 6, 20) match the built-in-system pipeline's defaults.
    lpa_n_panels:  int = 8    # number of LPA panels
    lpa_max_order: int = 6    # Legendre polynomial order per panel
    lpa_nt_seg:    int = 20   # collocation/output points per segment

    # Physics-residual loss is normalized per-state by (state_bounds span)/dt
    # by default (so e.g. a fast electrical state doesn't get swamped by a
    # slow thermal one purely by scale). Override per state name here,
    # e.g. {"SOC": 0.5}.
    res_scale: Optional[Dict[str, float]] = None
