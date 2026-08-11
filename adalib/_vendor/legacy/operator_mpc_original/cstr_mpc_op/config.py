#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
============================================================
Single library-wide config. Pick a problem via PROBLEM_NAME (or the env var
PROBLEM=<name>); all module-level constants below are then resolved from the
PROBLEM_CONFIGS table.

Problems registered:
    "lotka"       — Lotka-Volterra (U, R) parameterization, Hamiltonian conservation
    "bioreactor"  — Fed-batch bioreactor (Haldane kinetics)
    "cstr"        — CSTR benchmark (Arrhenius + heat balance, stiff → BDF)
    "triple_tank" — Three-tank Torricelli outflow

The rest of the library (models, data, learner, mains) reads only the
RESOLVED constants below — it does not touch PROBLEM_CONFIGS directly. To add
a new problem, register it here and create problems/<name>_problem.py.
"""

import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# Problem selection
# ============================================================
PROBLEM_NAME = os.environ.get("PROBLEM", "lotka").lower()

# ============================================================
# Basis selection
# ============================================================
# "lpa"  — W-panel Legendre (hard x(0)=x0)
# "adaf" — Fourier basis (hard x(0)=x0 AND ẋ(0)=f(x0, θ))
BASIS_NAME = os.environ.get("BASIS", "lpa").lower()

# ============================================================
# Library-wide constants (problem-independent)
# ============================================================
USE_GPU = 0
SEED = None               # None → fresh random seed per run
NP_DTYPE = np.float32
TF_DTYPE = "float32"

GAMMA = 1.0               # LPA basis time scale (NOT to be confused with LV's γ)

# ADA-F basis (only used when BASIS_NAME == "adaf")
# N_M  = Fourier mode truncation
# N_P  = collocation/panel count = network output width per state
ADAF_N_M = 40
ADAF_N_P = 40
GRAD_CLIP_NORM = 1.0
USE_JIT = True

# Optimizer / LR schedule
PHYSICS_LR = 3e-3
LR_MIN = 5e-5
USE_LR_SCHEDULE = True
LR_WARMUP_FRACTION = 0.05

# Rollout — DISABLED by default. Per project guidance the operator should
# learn one segment well across diverse ICs, and rollout is then an
# inference-only chaining of segments. K-segment chained gradient training is
# kept as an opt-in diagnostic; enable via `--rollout_epochs N`.
ROLLOUT_EPOCHS = 0
ROLLOUT_STEPS = 3
ROLLOUT_ALPHA = 0.75
ROLLOUT_LR = LR_MIN

# IO
DATA_DIR_ROOT = "data_files"
RESULT_DIR_ROOT = "results"


# ============================================================
# Per-problem configuration table
# ============================================================
PROBLEM_CONFIGS = {
    "lotka": {
        # time / segments
        "T0": 0.0, "T_FINAL": 1.0, "N_SEG": 20, "NT_SEG": 201,
        # LPA basis
        "MAX_ORDER": 10, "LPA_N_P": 21,
        # network
        "HIDDEN": 256, "N_LAYERS": 3,
        # training
        "PHYSICS_EPOCHS": 2000, "PHYSICS_BATCH_SIZE": 128,
        # data counts
        "N_TRAIN_CASES": 4000, "N_VAL_CASES": 800, "N_TEST_CASES": 800,
        # reference solver
        "REF_SOLVER": "RK45", "REF_RTOL": 1e-9, "REF_ATOL": 1e-11,
        # ODE structure
        "STATE_DIM": 2, "PARAM_DIM": 4, "RES_SCALE": (1.0, 1.0),
        # data prefix
        "DATA_PREFIX": "lotka",
    },
    "bioreactor": {
        "T0": 0.0, "T_FINAL": 100.0, "N_SEG": 50, "NT_SEG": 41,
        "MAX_ORDER": 4, "LPA_N_P": 30,
        "HIDDEN": 128, "N_LAYERS": 3,
        "PHYSICS_EPOCHS": 2000, "PHYSICS_BATCH_SIZE": 128,
        "N_TRAIN_CASES": 50000, "N_VAL_CASES": 10000, "N_TEST_CASES": 10000,
        "REF_SOLVER": "RK45", "REF_RTOL": 1e-9, "REF_ATOL": 1e-11,
        "STATE_DIM": 4, "PARAM_DIM": 3, "RES_SCALE": (0.033, 3.4, 0.025, 0.1),
        "DATA_PREFIX": "bioreactor",
    },
    "cstr": {
        # T_FINAL 1.0 → 0.5: CSTR converges to steady state by ~0.2–0.25h.
        # Truncating the long tail halves the rollout chain (less compounding),
        # halves training/inference cost, doubles plot information density.
        "T0": 0.0, "T_FINAL": 0.5, "N_SEG": 25, "NT_SEG": 41,
        "MAX_ORDER": 10, "LPA_N_P": 24,    # Step 1: high order for sharp Arrhenius
        "HIDDEN": 128, "N_LAYERS": 3,    # Step 3: wider net for non-linearity
        "PHYSICS_EPOCHS": 2000, "PHYSICS_BATCH_SIZE": 256,    # V3-B: 2000 epoch (user) + V3 batch 256
        # V3-B: case count 20000 → 2000 (V3 parity); paired with
        # `segment_sampling_strategy="all"` (CSTR builds all 50 segs/case →
        # 100K segments, then early-transient oversample → ~360K effective).
        "N_TRAIN_CASES": 2000, "N_VAL_CASES": 200, "N_TEST_CASES": 200,
        "REF_SOLVER": "BDF", "REF_RTOL": 1e-8, "REF_ATOL": 1e-10,    # CSTR is stiff
        "STATE_DIM": 4, "PARAM_DIM": 4,
        # Step 3: RES_SCALE rebalanced — old (1, 1, 100, 100) gave T residual
        # ~10000× weaker than C; new values match actual derivative magnitudes.
        "RES_SCALE": (0.5, 0.3, 30.0, 15.0),
        "DATA_PREFIX": "cstr",
    },
    "triple_tank": {
        "T0": 0.0, "T_FINAL": 300.0, "N_SEG": 10, "NT_SEG": 31,
        "MAX_ORDER": 5, "LPA_N_P": 20,
        "HIDDEN": 128, "N_LAYERS": 3,
        "PHYSICS_EPOCHS": 2000, "PHYSICS_BATCH_SIZE": 128,
        "N_TRAIN_CASES": 20000, "N_VAL_CASES": 4000, "N_TEST_CASES": 4000,
        "REF_SOLVER": "RK45", "REF_RTOL": 1e-9, "REF_ATOL": 1e-11,
        "STATE_DIM": 3, "PARAM_DIM": 2, "RES_SCALE": (1.0, 1.0, 1.0),
        "DATA_PREFIX": "triple_tank",
    },
    # ── CSTR MPC: Q 시퀀스를 입력으로 받는 MPC 전용 variant ──────────────
    # input: [C_A, C_B, T_R, T_K, Q_0, ..., Q_{N-1}]  (4 + N_SEG = 29D)
    # alpha=1, beta=1, F=50 고정. Q 시퀀스만 케이스마다 샘플링.
    # PARAM_DIM = N_SEG (Q 시퀀스 길이)
    "cstr_mpc": {
        "T0": 0.0, "T_FINAL": 0.5, "N_SEG": 25, "NT_SEG": 41,
        "MAX_ORDER": 10, "LPA_N_P": 24,
        "HIDDEN": 256, "N_LAYERS": 4,
        "PHYSICS_EPOCHS": 3000, "PHYSICS_BATCH_SIZE": 256,
        "N_TRAIN_CASES": 3000, "N_VAL_CASES": 300, "N_TEST_CASES": 300,
        "REF_SOLVER": "BDF", "REF_RTOL": 1e-8, "REF_ATOL": 1e-10,
        # segment 입력: [x_k(4), Q_k(1)] = 5D
        # PARAM_DIM=1 (Q_k scalar), INPUT_DIM=5
        "STATE_DIM": 4, "PARAM_DIM": 1,
        "RES_SCALE": (0.5, 0.3, 30.0, 15.0),
        "DATA_PREFIX": "cstr_mpc",
    },
    # ── Triple Tank MPC: [Q1, Q2] 시퀀스를 입력으로 받는 MPC 전용 variant ─
    # input: [h1, h2, h3, Q1_0,...,Q1_{N-1}, Q2_0,...,Q2_{N-1}]  (3 + N_SEG*2)
    # segment 입력: [h1_k, h2_k, h3_k, Q1_k, Q2_k] = 5D
    # PARAM_DIM=2 (Q1_k, Q2_k scalar 각 1개), INPUT_DIM=5
    "triple_tank_mpc": {
        "T0": 0.0, "T_FINAL": 600.0, "N_SEG": 50, "NT_SEG": 31,
        "MAX_ORDER": 5, "LPA_N_P": 20,
        "HIDDEN": 128, "N_LAYERS": 3,
        "PHYSICS_EPOCHS": 2000, "PHYSICS_BATCH_SIZE": 128,
        "N_TRAIN_CASES": 3000, "N_VAL_CASES": 300, "N_TEST_CASES": 300,
        "REF_SOLVER": "RK45", "REF_RTOL": 1e-9, "REF_ATOL": 1e-11,
        "STATE_DIM": 3, "PARAM_DIM": 2,
        "RES_SCALE": (1.0, 1.0, 1.0),
        "DATA_PREFIX": "triple_tank_mpc",
    },
}


def _resolve():
    if PROBLEM_NAME not in PROBLEM_CONFIGS:
        raise ValueError(
            f"Unknown PROBLEM_NAME={PROBLEM_NAME!r}. "
            f"Registered: {sorted(PROBLEM_CONFIGS)}"
        )
    return PROBLEM_CONFIGS[PROBLEM_NAME]


_cfg = _resolve()

# ----- time / segments -----
T0       = _cfg["T0"]
T_FINAL  = _cfg["T_FINAL"]
N_SEG    = _cfg["N_SEG"]
NT_SEG   = _cfg["NT_SEG"]
DT_SEG   = (T_FINAL - T0) / N_SEG
NT_TOTAL = N_SEG * (NT_SEG - 1) + 1

# ----- basis / net -----
MAX_ORDER = _cfg["MAX_ORDER"]
LPA_N_P   = _cfg["LPA_N_P"]
HIDDEN    = _cfg["HIDDEN"]
N_LAYERS  = _cfg["N_LAYERS"]

# ----- ODE shape -----
STATE_DIM = _cfg["STATE_DIM"]
PARAM_DIM = _cfg["PARAM_DIM"]
INPUT_DIM = STATE_DIM + PARAM_DIM
RES_SCALE = _cfg["RES_SCALE"]

# ----- training -----
PHYSICS_EPOCHS     = _cfg["PHYSICS_EPOCHS"]
PHYSICS_BATCH_SIZE = _cfg["PHYSICS_BATCH_SIZE"]

# ----- data -----
N_TRAIN_CASES = _cfg["N_TRAIN_CASES"]
N_VAL_CASES   = _cfg["N_VAL_CASES"]
N_TEST_CASES  = _cfg["N_TEST_CASES"]
DATA_PREFIX   = _cfg["DATA_PREFIX"]

# ----- reference solver -----
REF_SOLVER = _cfg["REF_SOLVER"]
REF_RTOL   = _cfg["REF_RTOL"]
REF_ATOL   = _cfg["REF_ATOL"]


def get_data_paths(problem_name=None):
    """Per-problem npz paths under data_files/<problem>/ ."""
    p = (problem_name or PROBLEM_NAME).lower()
    prefix = PROBLEM_CONFIGS[p]["DATA_PREFIX"]
    base = f"{DATA_DIR_ROOT}/{p}"
    return {
        "train_fullcase": f"{base}/{prefix}_train_fullcases.npz",
        "val_fullcase":   f"{base}/{prefix}_val_fullcases.npz",
        "test_fullcase":  f"{base}/{prefix}_test_fullcases.npz",
        "train_segment":  f"{base}/{prefix}_train_segments.npz",
        "val_segment":    f"{base}/{prefix}_val_segments.npz",
        "test_segment":   f"{base}/{prefix}_test_segments.npz",
    }


def get_result_dir(problem_name=None):
    """Per-problem result root: results/<problem>/ ."""
    p = (problem_name or PROBLEM_NAME).lower()
    return f"{RESULT_DIR_ROOT}/{p}"
