#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/dataset_builder.py
============================================================
Generic dataset builder + loader. Takes a Problem instance and produces:
  • <prefix>_train_fullcases.npz   (full RK45/BDF trajectories)
  • <prefix>_train_segments.npz    (one random segment per case)
plus val/test counterparts. Schema is identical across all problems.
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf

from config import (
    NP_DTYPE, T0, T_FINAL, N_SEG, NT_SEG,
)
from utils.io_utils import ensure_dir


# ============================================================
# Time grid
# ============================================================
def build_global_grid():
    dt_seg = (T_FINAL - T0) / N_SEG
    parts = []
    for k in range(N_SEG):
        t0 = T0 + k * dt_seg
        t1 = t0 + dt_seg
        local = np.linspace(t0, t1, NT_SEG, dtype=NP_DTYPE)
        if k > 0:
            local = local[1:]
        parts.append(local)
    return np.concatenate(parts, axis=0).astype(NP_DTYPE)


# ============================================================
# Full-case build
# ============================================================
def build_and_save_fullcase(problem, save_path, n_cases, seed):
    ensure_dir(os.path.dirname(save_path) or ".")
    t_grid = build_global_grid()
    x0, theta, meta = problem.sample_cases(n_cases, seed=seed)
    X0 = problem.build_input(x0, theta)

    Y = np.empty((x0.shape[0], t_grid.shape[0], problem.state_dim), dtype=NP_DTYPE)
    for i in range(x0.shape[0]):
        Y[i] = problem.solve_reference(theta[i], x0[i], t_grid)

    payload = dict(
        X0=X0, x0=x0.astype(NP_DTYPE), theta=theta.astype(NP_DTYPE),
        t_grid=t_grid, Y_ref=Y,
    )
    for k, v in (meta or {}).items():
        payload[k] = np.asarray(v, dtype=NP_DTYPE)
    np.savez_compressed(save_path, **payload)
    print(f"[SAVE] full-case -> {save_path}  X0={X0.shape}  Y_ref={Y.shape}")


# ============================================================
# Segment slicing
# ============================================================
def build_segments_from_fullcase(problem, fullcase_npz_path, seed=0):
    """Slice a fullcase npz into segment training rows.

    Strategy depends on `problem.segment_sampling_strategy`:
      "random" (default) — one random segment per case, 1:1 case↔sample ratio.
      "all"              — every one of the N_SEG segments per case, n_cases · N_SEG samples.
    """
    arr = np.load(fullcase_npz_path)
    X0 = arr["X0"].astype(NP_DTYPE)
    Y_ref = arr["Y_ref"].astype(NP_DTYPE)
    theta = arr["theta"].astype(NP_DTYPE)

    step = NT_SEG - 1
    t_local = np.linspace(0.0, (T_FINAL - T0) / N_SEG, NT_SEG, dtype=NP_DTYPE)
    n_cases = X0.shape[0]
    strategy = getattr(problem, "segment_sampling_strategy", "random")

    # uses_param_sequence=True인 문제(cstr_mpc, triple_tank_mpc 등):
    #   theta shape = (n_cases, N_SEG * param_dim)
    #   [param0_0,...,param0_{N-1}, param1_0,...,param1_{N-1}, ...]  (concatenated)
    #   세그먼트 k의 param i → theta[case_id, i*N_SEG + k]
    # 일반 문제: seg_X = [x_k, theta_flat] (state_dim + param_dim)
    uses_seq = getattr(problem, "uses_param_sequence", False)

    def _seq_seg_params(case_ids, seg_ids):
        """uses_seq=True일 때 세그먼트별 param 벡터 추출 → (n_samples, param_dim)."""
        pd = problem.param_dim
        col_idx = np.stack(
            [i * N_SEG + seg_ids for i in range(pd)], axis=-1
        ).astype(np.int64)   # (n_samples, pd)
        return theta[case_ids[:, None], col_idx].astype(NP_DTYPE)

    if strategy == "all":
        # All N_SEG segments per case → n_cases · N_SEG total rows.
        case_ids = np.repeat(np.arange(n_cases, dtype=np.int32), N_SEG)
        seg_ids = np.tile(np.arange(N_SEG, dtype=np.int32), n_cases)
        starts = seg_ids.astype(np.int64) * step
        seg_Y = np.stack([Y_ref[c, s:s + NT_SEG]
                          for c, s in zip(case_ids, starts)], axis=0)
        xk = seg_Y[:, 0, :problem.state_dim]
        if uses_seq:
            seg_X = np.concatenate([xk, _seq_seg_params(case_ids, seg_ids)], axis=-1).astype(NP_DTYPE)
        else:
            seg_X = np.concatenate([xk, theta[case_ids]], axis=-1).astype(NP_DTYPE)
    else:
        # Default: one random segment per case (1:1 case↔sample ratio).
        rng = np.random.default_rng(int(seed))
        seg_ids = rng.integers(0, N_SEG, size=n_cases).astype(np.int32)
        case_ids = np.arange(n_cases, dtype=np.int32)
        starts = seg_ids.astype(np.int64) * step
        seg_Y = np.stack([Y_ref[i, s:s + NT_SEG]
                          for i, s in zip(case_ids, starts)], axis=0)
        xk = seg_Y[:, 0, :problem.state_dim]
        if uses_seq:
            seg_X = np.concatenate([xk, _seq_seg_params(case_ids, seg_ids)], axis=-1).astype(NP_DTYPE)
        else:
            seg_X = np.concatenate([xk, theta[case_ids]], axis=-1).astype(NP_DTYPE)

    return {
        "X": seg_X,
        "Y_ref_seg": seg_Y,
        "case_id": case_ids,
        "seg_id": seg_ids,
        "t_local": t_local,
    }


def compute_input_stats(X):
    X_mean = X.mean(axis=0).astype(NP_DTYPE)
    X_std = np.maximum(X.std(axis=0), 1e-6).astype(NP_DTYPE)
    return X_mean, X_std


def save_segments(save_path, seg_dict, X_mean=None, X_std=None):
    ensure_dir(os.path.dirname(save_path) or ".")
    payload = dict(seg_dict)
    if X_mean is not None:
        payload["X_mean"] = np.asarray(X_mean, dtype=NP_DTYPE)
    if X_std is not None:
        payload["X_std"] = np.asarray(X_std, dtype=NP_DTYPE)
    np.savez_compressed(save_path, **payload)
    print(f"[SAVE] segments -> {save_path}  X={payload['X'].shape}  Y={payload['Y_ref_seg'].shape}")


# ============================================================
# Loaders
# ============================================================
def load_segments(npz_path):
    arr = np.load(npz_path)
    out = {
        "X": arr["X"].astype(np.float32),
        "Y_ref_seg": arr["Y_ref_seg"].astype(np.float32),
        "case_id": arr["case_id"].astype(np.int32),
        "seg_id": arr["seg_id"].astype(np.int32),
        "t_local": arr["t_local"].astype(np.float32),
    }
    if "X_mean" in arr:
        out["X_mean"] = arr["X_mean"].astype(np.float32)
    if "X_std" in arr:
        out["X_std"] = arr["X_std"].astype(np.float32)
    return out


def load_fullcase(npz_path):
    arr = np.load(npz_path)
    return {k: arr[k] for k in arr.files}


def make_tf_dataset(seg, batch_size=256, shuffle=True, problem=None):
    """Wrap a loaded segment dict as tf.data. When `shuffle=True` (= train
    split) and a problem is provided, the problem's `apply_train_oversampling`
    hook may duplicate rows (used by CSTR for early-transient oversampling)."""
    if shuffle and problem is not None:
        seg = problem.apply_train_oversampling(seg)
    ds = tf.data.Dataset.from_tensor_slices((seg["X"], seg["Y_ref_seg"]))
    if shuffle:
        ds = ds.shuffle(min(len(seg["X"]), 200000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
