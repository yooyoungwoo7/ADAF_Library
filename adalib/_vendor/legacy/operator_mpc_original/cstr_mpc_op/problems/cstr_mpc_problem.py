#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/cstr_mpc_problem.py
============================================================
CSTR MPC variant.

핵심 변경사항 (기존 cstr_problem.py 대비):
  - alpha, beta, F 고정 (single operating point)
  - Q를 스칼라 1개 → N_SEG개 시퀀스로 샘플링
  - Operator 입력: z_k = [C_A, C_B, T_R, T_K, Q_0, ..., Q_{N-1}]
                        = [state(4) + Q_seq(N_SEG)]
  - solve_reference(): 세그먼트 k마다 Q_seq[k]로 ODE 적분

MPC inference 시:
  - optimizer가 Q_seq를 최적화 변수로 사용
  - 각 세그먼트마다 rollout하며 T_R → T_ref 수렴
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

from config import NP_DTYPE, TF_DTYPE, N_SEG, T0, T_FINAL, NT_SEG
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)

# ── Arrhenius / heat-balance constants (동일) ──────────────
K0_AB   = 1.287e12
K0_BC   = 1.287e12
K0_AD   = 9.043e9
E_A_AB  = 9758.3
E_A_BC  = 9758.3
E_A_AD  = 8560.0
H_R_AB  = 4.2
H_R_BC  = -11.0
H_R_AD  = -41.85
RHO     = 0.9342
CP      = 3.01
CP_K    = 2.0
K_W     = 4032.0
A_R     = 0.215
V_R     = 10.01
M_K     = 5.0
T_IN    = 130.0
C_A0_FEED = (5.7 + 4.5) / 2.0

# ── 고정 operating point ───────────────────────────────────
ALPHA_FIXED = 1.0
BETA_FIXED  = 1.0
F_FIXED     = 50.0   # dilution rate [1/h]

Q_RANGE = (-8500.0, 0.0)   # heat duty [kJ/h], MPC 최적화 범위


class CSTRMPCProblem(BaseProblem):
    """
    CSTR with Q-sequence input for MPC.

    param_dim = N_SEG  (Q_0, Q_1, ..., Q_{N-1})
    input_dim = state_dim(4) + N_SEG
    """

    name   = "cstr_mpc"
    state_dim = 4
    # param_dim은 N_SEG에 따라 동적으로 결정 (property로 처리)
    state_labels = ("C_A", "C_B", "T_R", "T_K")
    time_unit    = "min"
    time_factor  = 60.0
    ref_solver   = "BDF"
    ref_rtol     = 1e-8
    ref_atol     = 1e-10
    segment_sampling_strategy = "all"
    uses_param_sequence = True   # dataset_builder에 per-segment 슬라이싱 신호

    X0_LOWER   = np.array([0.1, 0.1,  50.0,  50.0], dtype=NP_DTYPE)
    X0_UPPER   = np.array([2.0, 2.0, 140.0, 140.0], dtype=NP_DTYPE)
    X0_NOMINAL = np.array([0.8, 0.5, 134.14, 130.0], dtype=NP_DTYPE)
    X0_PERTURB_FRAC = 0.10   # I.C 다양성을 위해 약간 넓게

    # derived features: k1, k2, k3 at T_R (alpha/beta 고정이므로 단순화)
    n_derived_features = 3
    derived_mean = np.array([52.0, 52.0, 13.0], dtype=NP_DTYPE)
    derived_std  = np.array([30.0, 30.0, 14.0], dtype=NP_DTYPE)

    output_scale = np.array([0.5, 0.3, 30.0, 15.0], dtype=NP_DTYPE)

    @property
    def param_dim(self):
        # segment 입력 기준: Q_k scalar 1개
        return 1

    @property
    def input_dim(self):
        # [x_k(4), Q_k(1)] = 5D
        return self.state_dim + 1

    # ── sampling ──────────────────────────────────────────
    def sample_cases(self, n, seed):
        """
        - alpha, beta, F: 고정
        - Q: 케이스마다 N_SEG개 독립 샘플링
        - x0: nominal 주변 perturb
        theta shape: (n, N_SEG)  ← Q 시퀀스
        """
        rng = np.random.default_rng(seed)

        # Q 시퀀스: 케이스 × 세그먼트
        Q_seq = rng.uniform(
            Q_RANGE[0], Q_RANGE[1],
            size=(n, N_SEG),
        ).astype(NP_DTYPE)

        # x0 perturbation
        noise = rng.uniform(-1.0, 1.0, size=(n, self.state_dim)).astype(NP_DTYPE)
        span  = self.X0_PERTURB_FRAC * (self.X0_UPPER - self.X0_LOWER)
        x0    = self.X0_NOMINAL[None, :] + noise * span[None, :]
        x0    = np.clip(x0, self.X0_LOWER, self.X0_UPPER).astype(NP_DTYPE)

        # theta = Q 시퀀스 (N_SEG columns)
        return x0, Q_seq, {}

    # ── build_input override ───────────────────────────────
    def build_input(self, x0, theta):
        """
        전체 케이스 입력 X0 = [x0, Q_seq]
        x0:    (state_dim,)  또는 (n, state_dim)
        theta: (N_SEG,)      또는 (n, N_SEG)
        """
        x0    = np.asarray(x0,    dtype=NP_DTYPE)
        theta = np.asarray(theta,  dtype=NP_DTYPE)
        if x0.ndim == 1:
            return np.concatenate([x0, theta], axis=0)
        return np.concatenate([x0, theta], axis=-1)

    # ── ODE RHS (numpy) ───────────────────────────────────
    @staticmethod
    def _rhs(t, x, Q):
        C_A, C_B, T_R, T_K = np.asarray(x, dtype=np.float64)
        Q = float(Q)
        Tc = T_R + 273.15
        k1 = BETA_FIXED  * K0_AB * np.exp(-E_A_AB / Tc)
        k2 =               K0_BC * np.exp(-E_A_BC / Tc)
        k3 = K0_AD * np.exp(-ALPHA_FIXED * E_A_AD / Tc)
        dC_A = F_FIXED * (C_A0_FEED - C_A) - k1 * C_A - k3 * C_A * C_A
        dC_B = -F_FIXED * C_B + k1 * C_A - k2 * C_B
        rh   = k1*C_A*H_R_AB + k2*C_B*H_R_BC + k3*C_A*C_A*H_R_AD
        dT_R = (rh / (-RHO * CP)
                + F_FIXED * (T_IN - T_R)
                + K_W * A_R * (T_K - T_R) / (RHO * CP * V_R))
        dT_K = (Q + K_W * A_R * (T_R - T_K)) / (M_K * CP_K)
        return np.array([dC_A, dC_B, dT_R, dT_K], dtype=np.float64)

    # ── solve_reference: 세그먼트별 Q_k로 적분 ─────────────
    def solve_reference(self, theta, x0, t_grid):
        """
        theta: (N_SEG,) — Q 시퀀스
        각 세그먼트 k에서 Q_seq[k]로 ODE 적분.
        """
        Q_seq    = np.asarray(theta, dtype=np.float64)
        dt_seg   = (T_FINAL - T0) / N_SEG
        xk       = np.asarray(x0, dtype=np.float64)
        all_rows = []

        for k in range(N_SEG):
            t0_k = T0 + k * dt_seg
            t1_k = t0_k + dt_seg
            # t_grid에서 이 세그먼트 구간 추출
            mask   = (t_grid >= t0_k - 1e-12) & (t_grid <= t1_k + 1e-12)
            t_eval = t_grid[mask].astype(np.float64)

            sol = solve_ivp(
                fun    = lambda t, y: self._rhs(t, y, Q_seq[k]),
                t_span = (float(t_eval[0]), float(t_eval[-1])),
                y0     = xk,
                t_eval = t_eval,
                method = self.ref_solver,
                rtol   = self.ref_rtol,
                atol   = self.ref_atol,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed seg {k}: {sol.message}")

            rows = sol.y.T.astype(NP_DTYPE)   # (NT_SEG, 4)
            if k == 0:
                all_rows.append(rows)
            else:
                all_rows.append(rows[1:])     # 중복 제거

            xk = sol.y[:, -1].copy()

        return np.concatenate(all_rows, axis=0)

    # ── rhs_tf: 현재 세그먼트 Q 사용 ─────────────────────
    def rhs_tf(self, x, theta):
        """
        theta의 마지막 원소를 현재 세그먼트 Q로 사용.
        dataset_builder에서 seg_X = [x_k, Q_seq] 전달 시
        learner가 theta[..., seg_id]를 꺼내거나,
        단순화: theta[..., 0]을 현재 Q로 약속.
        → dataset_builder에서 seg_X = [x_k, Q_k_scalar] (5D)로 슬라이스.
        """
        x     = tf.convert_to_tensor(x,     dtype=DTYPE)
        theta = tf.convert_to_tensor(theta,  dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)

        C_A = x[..., 0]; C_B = x[..., 1]
        T_R = x[..., 2]; T_K = x[..., 3]
        Q   = theta[..., 0]   # 현재 세그먼트 Q_k

        Tc   = T_R + tf.constant(273.15, dtype=DTYPE)
        inv_T = 1.0 / Tc
        k1 = BETA_FIXED  * K0_AB * tf.exp(-E_A_AB * inv_T)
        k2 =               K0_BC * tf.exp(-E_A_BC * inv_T)
        k3 =               K0_AD * tf.exp(-ALPHA_FIXED * E_A_AD * inv_T)

        dC_A = F_FIXED * (C_A0_FEED - C_A) - k1*C_A - k3*tf.square(C_A)
        dC_B = -F_FIXED*C_B + k1*C_A - k2*C_B
        rh   = k1*C_A*H_R_AB + k2*C_B*H_R_BC + k3*tf.square(C_A)*H_R_AD
        dT_R = (rh / (-RHO*CP)
                + F_FIXED*(T_IN - T_R)
                + K_W*A_R*(T_K - T_R) / (RHO*CP*V_R))
        dT_K = (Q + K_W*A_R*(T_R - T_K)) / (M_K*CP_K)
        return tf.stack([dC_A, dC_B, dT_R, dT_K], axis=-1)

    # ── derived features ─────────────────────────────────
    def derived_features_tf(self, x_input):
        """k1, k2, k3 at current T_R. alpha/beta/F는 고정이므로 불필요."""
        x   = tf.convert_to_tensor(x_input, dtype=DTYPE)
        T_R = x[:, 2]
        inv_T = 1.0 / (T_R + tf.constant(273.15, dtype=DTYPE))
        k1 = BETA_FIXED  * K0_AB * tf.exp(-E_A_AB * inv_T)
        k2 =               K0_BC * tf.exp(-E_A_BC * inv_T)
        k3 =               K0_AD * tf.exp(-ALPHA_FIXED * E_A_AD * inv_T)
        return tf.stack([k1, k2, k3], axis=-1)

    # ── split_input_tf ────────────────────────────────────
    def split_input_tf(self, X, dtype=tf.float32):
        X = tf.convert_to_tensor(X, dtype=dtype)
        x0    = X[..., :self.state_dim]
        theta = X[..., self.state_dim:]   # Q 시퀀스 (N_SEG,)
        return x0, theta

    # ── nominal_input ─────────────────────────────────────
    def nominal_input(self):
        Q_nom = np.full((N_SEG,), -2000.0, dtype=NP_DTYPE)
        return np.concatenate([self.X0_NOMINAL, Q_nom])

    def case_subtitle(self, x_input):
        C_A, C_B, T_R, T_K = x_input[:4]
        Q_seq = x_input[4:]
        return (f"$C_A$={C_A:.2f}, $C_B$={C_B:.2f}\n"
                f"$T_R$={T_R:.1f}, $T_K$={T_K:.1f}\n"
                f"mean Q={Q_seq.mean():.0f} kJ/h")

    def state_units(self):
        return ("mol/l", "mol/l", "°C", "°C")

    def state_plot_labels(self):
        return (r"$C_A$", r"$C_B$", r"$T_R$", r"$T_K$")
