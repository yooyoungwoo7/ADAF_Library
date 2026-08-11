#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/triple_tank_mpc_problem.py
============================================================
Triple Tank MPC variant.

핵심 변경사항 (기존 triple_tank_problem.py 대비):
  - Q1, Q2를 스칼라가 아닌 N_SEG개 시퀀스로 샘플링
  - Operator 입력: z_k = [h1_k, h2_k, h3_k, Q1_k, Q2_k]  → 5D
  - theta shape: (n_cases, N_SEG*2)
        앞 N_SEG개 = Q1 시퀀스,  뒤 N_SEG개 = Q2 시퀀스
  - solve_reference(): 세그먼트 k마다 (Q1_seq[k], Q2_seq[k])로 ODE 적분
  - 제어변수: h3 (index=2)

MPC inference 시:
  - optimizer가 [Q1_k, Q2_k]를 최적화 변수로 사용
  - 각 세그먼트마다 rollout하며 h3 → h3_target 수렴
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

from config import NP_DTYPE, TF_DTYPE, N_SEG, T0, T_FINAL
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)

# ODE constants (triple_tank_problem.py와 동일)
A_TANK      = 154.0
S_N         = 0.5
G_ACC       = 981.0
A1          = 0.46
A2          = 0.60
A3          = 0.45
H_MIN_FLOOR = 0.0
SQRT_EPS    = 1e-8

Q1_RANGE = (0.0, 100.0)
Q2_RANGE = (0.0, 100.0)


class TripleTankMPCProblem(BaseProblem):
    """
    Triple Tank with [Q1, Q2]-sequence input for MPC.

    param_dim = 2  (Q1_k, Q2_k scalar per segment)
    input_dim = 5  (state_dim=3 + param_dim=2)
    theta storage per case: (N_SEG*2,)
        [Q1_0,...,Q1_{N-1}, Q2_0,...,Q2_{N-1}]
    """

    name      = "triple_tank_mpc"
    state_dim = 3
    state_labels  = ("h1", "h2", "h3")
    time_unit     = "s"
    time_factor   = 1.0
    ref_solver    = "RK45"
    ref_rtol      = 1e-9
    ref_atol      = 1e-11
    segment_sampling_strategy = "random"
    uses_param_sequence = True   # dataset_builder에 per-segment 슬라이싱 신호

    X0_LOWER = np.array([  5.0,   5.0,   5.0], dtype=NP_DTYPE)
    X0_UPPER = np.array([300.0, 300.0, 300.0], dtype=NP_DTYPE)

    # Torricelli 유동 6개 derived features (새 스케일: h~150 cm, Q~50 cm³/s)
    # Q13≈50, Q32≈50, Q20≈100, net≈0
    n_derived_features = 6
    derived_mean = np.array([50.0, 50.0, 100.0, 0.0, 0.0, 0.0], dtype=NP_DTYPE)
    derived_std  = np.array([40.0, 40.0,  60.0, 70.0, 70.0, 50.0], dtype=NP_DTYPE)

    @property
    def param_dim(self):
        return 2

    @property
    def input_dim(self):
        return self.state_dim + 2   # 5D

    # ── sampling ──────────────────────────────────────────
    def sample_cases(self, n, seed):
        """
        theta shape: (n, N_SEG*2)
            앞 N_SEG열 = Q1 시퀀스,  뒤 N_SEG열 = Q2 시퀀스
        """
        rng = np.random.default_rng(seed)

        h1 = rng.uniform(float(self.X0_LOWER[0]), float(self.X0_UPPER[0]), size=(n,))
        h2 = rng.uniform(float(self.X0_LOWER[1]), float(self.X0_UPPER[1]), size=(n,))
        h3 = rng.uniform(float(self.X0_LOWER[2]), float(self.X0_UPPER[2]), size=(n,))
        x0 = np.stack([h1, h2, h3], axis=-1).astype(NP_DTYPE)

        Q1_seq = rng.uniform(*Q1_RANGE, size=(n, N_SEG)).astype(NP_DTYPE)
        Q2_seq = rng.uniform(*Q2_RANGE, size=(n, N_SEG)).astype(NP_DTYPE)
        theta  = np.concatenate([Q1_seq, Q2_seq], axis=-1)  # (n, N_SEG*2)

        return x0, theta, {}

    # ── build_input ───────────────────────────────────────
    def build_input(self, x0, theta):
        x0    = np.asarray(x0,    dtype=NP_DTYPE)
        theta = np.asarray(theta, dtype=NP_DTYPE)
        if x0.ndim == 1:
            return np.concatenate([x0, theta], axis=0)
        return np.concatenate([x0, theta], axis=-1)

    # ── ODE RHS (numpy) ───────────────────────────────────
    def rhs_np(self, t, x, theta):
        """BaseProblem-compatible RHS. theta = [Q1, Q2] (current segment)."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        return self._rhs(t, x, theta[0], theta[1])

    @staticmethod
    def _rhs(t, x, Q1, Q2):
        h1, h2, h3 = np.asarray(x, dtype=np.float64)
        Q1, Q2 = float(Q1), float(Q2)
        h1p = max(h1 - H_MIN_FLOOR, 0.0)
        h2p = max(h2 - H_MIN_FLOOR, 0.0)
        h3p = max(h3 - H_MIN_FLOOR, 0.0)
        dh13 = h1p - h3p
        dh32 = h3p - h2p
        Q13 = A1 * S_N * np.sign(dh13) * np.sqrt(2.0 * G_ACC * abs(dh13) + SQRT_EPS)
        Q32 = A3 * S_N * np.sign(dh32) * np.sqrt(2.0 * G_ACC * abs(dh32) + SQRT_EPS)
        Q20 = A2 * S_N * np.sqrt(2.0 * G_ACC * h2p + SQRT_EPS)
        return np.array([
            (Q1 - Q13) / A_TANK,
            (Q2 + Q32 - Q20) / A_TANK,
            (Q13 - Q32) / A_TANK,
        ], dtype=np.float64)

    # ── solve_reference ───────────────────────────────────
    def solve_reference(self, theta, x0, t_grid):
        """
        theta: (N_SEG*2,) — [Q1_0,...,Q1_{N-1}, Q2_0,...,Q2_{N-1}]
        세그먼트 k마다 (Q1_seq[k], Q2_seq[k])로 ODE 적분.
        """
        theta  = np.asarray(theta, dtype=np.float64)
        Q1_seq = theta[:N_SEG]
        Q2_seq = theta[N_SEG:]
        dt_seg = (T_FINAL - T0) / N_SEG
        xk     = np.asarray(x0, dtype=np.float64)
        all_rows = []

        for k in range(N_SEG):
            t0_k = T0 + k * dt_seg
            t1_k = t0_k + dt_seg
            mask   = (t_grid >= t0_k - 1e-12) & (t_grid <= t1_k + 1e-12)
            t_eval = t_grid[mask].astype(np.float64)

            sol = solve_ivp(
                fun    = lambda t, y: self._rhs(t, y, Q1_seq[k], Q2_seq[k]),
                t_span = (float(t_eval[0]), float(t_eval[-1])),
                y0     = xk,
                t_eval = t_eval,
                method = self.ref_solver,
                rtol   = self.ref_rtol,
                atol   = self.ref_atol,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed seg {k}: {sol.message}")

            rows = sol.y.T.astype(NP_DTYPE)
            if k == 0:
                all_rows.append(rows)
            else:
                all_rows.append(rows[1:])   # 중복 끝점 제거
            xk = sol.y[:, -1].copy()

        return np.concatenate(all_rows, axis=0)

    # ── rhs_tf ────────────────────────────────────────────
    def rhs_tf(self, x, theta):
        """theta[...,0]=Q1_k, theta[...,1]=Q2_k (현재 세그먼트 입력)."""
        x     = tf.convert_to_tensor(x,     dtype=DTYPE)
        theta = tf.convert_to_tensor(theta,  dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)

        h1 = tf.nn.relu(x[..., 0] - H_MIN_FLOOR)
        h2 = tf.nn.relu(x[..., 1] - H_MIN_FLOOR)
        h3 = tf.nn.relu(x[..., 2] - H_MIN_FLOOR)
        Q1 = theta[..., 0]
        Q2 = theta[..., 1]

        g   = tf.constant(G_ACC,    dtype=DTYPE)
        sn  = tf.constant(S_N,      dtype=DTYPE)
        A_  = tf.constant(A_TANK,   dtype=DTYPE)
        a1  = tf.constant(A1,       dtype=DTYPE)
        a2  = tf.constant(A2,       dtype=DTYPE)
        a3  = tf.constant(A3,       dtype=DTYPE)
        eps = tf.constant(SQRT_EPS, dtype=DTYPE)

        dh13 = h1 - h3
        Q13  = a1 * sn * tf.sign(dh13) * tf.sqrt(2.0 * g * tf.abs(dh13) + eps)
        dh32 = h3 - h2
        Q32  = a3 * sn * tf.sign(dh32) * tf.sqrt(2.0 * g * tf.abs(dh32) + eps)
        Q20  = a2 * sn * tf.sqrt(2.0 * g * h2 + eps)

        Q1_b = tf.broadcast_to(Q1, tf.shape(h1))
        Q2_b = tf.broadcast_to(Q2, tf.shape(h2))

        dh1 = (Q1_b - Q13) / A_
        dh2 = (Q2_b + Q32 - Q20) / A_
        dh3 = (Q13 - Q32) / A_
        return tf.stack([dh1, dh2, dh3], axis=-1)

    # ── derived_features_tf ───────────────────────────────
    def derived_features_tf(self, x_input):
        """6 Torricelli flow features from [h1, h2, h3, Q1_k, Q2_k] (triple_tank_problem.py와 동일)."""
        x = tf.convert_to_tensor(x_input, dtype=DTYPE)
        h1 = tf.nn.relu(x[:, 0] - H_MIN_FLOOR)
        h2 = tf.nn.relu(x[:, 1] - H_MIN_FLOOR)
        h3 = tf.nn.relu(x[:, 2] - H_MIN_FLOOR)
        Q1 = x[:, 3]
        Q2 = x[:, 4]

        eps = tf.constant(SQRT_EPS, dtype=DTYPE)
        g   = tf.constant(G_ACC,    dtype=DTYPE)
        sn  = tf.constant(S_N,      dtype=DTYPE)
        a1  = tf.constant(A1,       dtype=DTYPE)
        a2  = tf.constant(A2,       dtype=DTYPE)
        a3  = tf.constant(A3,       dtype=DTYPE)

        dh13 = h1 - h3
        Q13  = a1 * sn * tf.sign(dh13) * tf.sqrt(2.0 * g * tf.abs(dh13) + eps)
        dh32 = h3 - h2
        Q32  = a3 * sn * tf.sign(dh32) * tf.sqrt(2.0 * g * tf.abs(dh32) + eps)
        Q20  = a2 * sn * tf.sqrt(2.0 * g * h2 + eps)
        net1 = Q1 - Q13
        net2 = Q2 + Q32 - Q20
        net3 = Q13 - Q32
        return tf.stack([Q13, Q32, Q20, net1, net2, net3], axis=-1)

    # ── split_input_tf ────────────────────────────────────
    def split_input_tf(self, X, dtype=tf.float32):
        X = tf.convert_to_tensor(X, dtype=dtype)
        x0    = X[..., :self.state_dim]
        theta = X[..., self.state_dim:]   # Q 시퀀스 (N_SEG*2,)
        return x0, theta

    # ── nominal_input ─────────────────────────────────────
    def nominal_input(self):
        # Q=67.7 cm³/s → h1≈194, h2≈104, h3≈150 cm 정상상태
        Q_nom = np.full((N_SEG,), 68.0, dtype=NP_DTYPE)
        return np.concatenate([
            np.array([194.0, 104.0, 150.0], dtype=NP_DTYPE),
            Q_nom, Q_nom.copy(),
        ])

    def case_subtitle(self, x_input):
        h1, h2, h3 = x_input[:3]
        Q1_seq = x_input[3:3 + N_SEG]
        Q2_seq = x_input[3 + N_SEG:]
        return (f"$h_1$={h1:.1f}, $h_2$={h2:.1f}, $h_3$={h3:.1f}\n"
                f"mean Q1={Q1_seq.mean():.0f}, mean Q2={Q2_seq.mean():.0f} cm³/s")

    def state_plot_labels(self):
        return (r"$h_1$ [cm]", r"$h_2$ [cm]", r"$h_3$ [cm]")

    def state_units(self):
        return ("cm", "cm", "cm")
