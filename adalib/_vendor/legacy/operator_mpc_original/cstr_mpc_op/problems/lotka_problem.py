#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/lotka_problem.py
============================================================
Classical Lotka-Volterra predator-prey, v2-aligned (U, R) parameterization:

    α = 2·R          β = 0.04·R·U
    γ = 1.06·R       δ = 0.02·R·U
    r₀ = 100/U       p₀ = 15/U

ODE:
    ṙ = α r − β r p
    ṗ = δ r p − γ p

Conservation: H = δ r − γ ln r + β p − α ln p (LV Hamiltonian).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from config import NP_DTYPE, TF_DTYPE, DT_SEG
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)
S_MIN_FLOOR = 0.0
LOG_FLOOR = 1e-6


class LotkaProblem(BaseProblem):
    name = "lotka"
    state_dim = 2
    param_dim = 4
    state_labels = ("r", "p")
    param_labels = ("alpha", "beta", "gamma", "delta")
    time_unit = ""

    def state_plot_labels(self):
        return (r"$r$", r"$p$")

    cons_w = 0.05

    # Physics-informed derived features (v2-style):
    #   ln(r₀/r*), ln(p₀/p*), V₀ (Hamiltonian),
    #   ω·T_seg, growth_r = α − β p₀, growth_p = δ r₀ − γ
    # Norm constants are tuned on the (U, R) family — V₀ ≈ 130–200, growth ≈ 20.
    n_derived_features = 6
    derived_mean = np.array([0.7, -1.1, 160.0, 1.4, 30.0, 20.0], dtype=NP_DTYPE)
    derived_std  = np.array([0.5,  0.5,  40.0, 0.3, 10.0, 10.0], dtype=NP_DTYPE)

    U_RANGE = (180.0, 220.0)
    R_RANGE = (15.0, 25.0)

    # ---------------------------------------------------------------
    @staticmethod
    def theta_from_UR(U, R):
        U = np.asarray(U, dtype=NP_DTYPE)
        R = np.asarray(R, dtype=NP_DTYPE)
        alpha = 2.0 * R
        beta  = 0.04 * R * U
        gamma = 1.06 * R
        delta = 0.02 * R * U
        return np.stack([alpha, beta, gamma, delta], axis=-1).astype(NP_DTYPE)

    @staticmethod
    def initial_state_from_U(U):
        U = np.asarray(U, dtype=NP_DTYPE)
        r0 = 100.0 / U
        p0 = 15.0 / U
        return np.stack([r0, p0], axis=-1).astype(NP_DTYPE)

    # ---------------------------------------------------------------
    def sample_cases(self, n, seed):
        rng = np.random.default_rng(seed)
        U = rng.uniform(*self.U_RANGE, size=(n,)).astype(NP_DTYPE)
        R = rng.uniform(*self.R_RANGE, size=(n,)).astype(NP_DTYPE)
        x0 = self.initial_state_from_U(U)
        theta = self.theta_from_UR(U, R)
        return x0, theta, {"U": U, "R": R}

    def rhs_np(self, t, x, theta):
        r, p = np.asarray(x, dtype=np.float64)
        alpha, beta, gamma, delta = np.asarray(theta, dtype=np.float64)
        rp = max(r, S_MIN_FLOOR)
        pp = max(p, S_MIN_FLOOR)
        return np.array([
            alpha * rp - beta  * rp * pp,
            delta * rp * pp - gamma * pp,
        ], dtype=np.float64)

    def rhs_tf(self, x, theta):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        theta = tf.convert_to_tensor(theta, dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)
        r, p = x[..., 0], x[..., 1]
        alpha = theta[..., 0]
        beta  = theta[..., 1]
        gamma = theta[..., 2]
        delta = theta[..., 3]
        floor = tf.constant(S_MIN_FLOOR, dtype=DTYPE)
        rp = tf.maximum(r, floor)
        pp = tf.maximum(p, floor)
        dr = alpha * rp - beta  * rp * pp
        dp = delta * rp * pp - gamma * pp
        return tf.stack([dr, dp], axis=-1)

    def derived_features_tf(self, x_input):
        """6 LV-specific physics features from raw input [r₀, p₀, α, β, γ, δ]."""
        x = tf.convert_to_tensor(x_input, dtype=DTYPE)
        eps = tf.constant(LOG_FLOOR, dtype=DTYPE)
        r0    = tf.maximum(x[:, 0], eps)
        p0    = tf.maximum(x[:, 1], eps)
        alpha = x[:, 2]
        beta  = tf.maximum(x[:, 3], eps)
        gamma = x[:, 4]
        delta = tf.maximum(x[:, 5], eps)

        r_star = gamma / delta
        p_star = alpha / beta
        ln_r = tf.math.log(r0 / r_star)
        ln_p = tf.math.log(p0 / p_star)
        V0 = (delta * r0 - gamma * tf.math.log(r0)
              + beta  * p0 - alpha * tf.math.log(p0))
        omega_T = tf.sqrt(tf.maximum(alpha * gamma, eps)) * tf.constant(float(DT_SEG), dtype=DTYPE)
        growth_r = alpha - beta * p0
        growth_p = delta * r0 - gamma
        return tf.stack([ln_r, ln_p, V0, omega_T, growth_r, growth_p], axis=-1)

    def conservation_quantity(self, x, theta):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        theta = tf.convert_to_tensor(theta, dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)
        floor = tf.constant(LOG_FLOOR, dtype=DTYPE)
        rp = tf.maximum(x[..., 0], floor)
        pp = tf.maximum(x[..., 1], floor)
        alpha = theta[..., 0]
        beta  = theta[..., 1]
        gamma = theta[..., 2]
        delta = theta[..., 3]
        return delta * rp - gamma * tf.math.log(rp) + beta * pp - alpha * tf.math.log(pp)

    # ---------------------------------------------------------------
    # plotting metadata
    # ---------------------------------------------------------------
    def nominal_input(self):
        U = 0.5 * (self.U_RANGE[0] + self.U_RANGE[1])
        R = 0.5 * (self.R_RANGE[0] + self.R_RANGE[1])
        x0 = self.initial_state_from_U(U)
        th = self.theta_from_UR(U, R)
        return np.concatenate([x0, th], axis=0).astype(NP_DTYPE)

    def sweep_specs(self):
        # Sweep U with R fixed and vice versa — but since both r0/p0 and θ
        # depend on (U, R), the input idx-based sweep doesn't make sense.
        # Provide nominal vs. sample directly via random_input below; sweeps
        # are handled specially in sweep_plots.lotka_sweep.
        return []

    def random_input(self, rng):
        U = float(rng.uniform(*self.U_RANGE))
        R = float(rng.uniform(*self.R_RANGE))
        x0 = self.initial_state_from_U(U).reshape(-1)
        th = self.theta_from_UR(U, R).reshape(-1)
        return np.concatenate([x0, th]).astype(NP_DTYPE)

    # 4 (U, R) archetypes spanning slow→fast oscillators with training-friendly
    # values (well inside U_RANGE × R_RANGE):
    _ARCHETYPES_UR = [
        (185, 16),   # slow + low U   → ~3.7 cycles, large amplitude
        (200, 20),   # canonical mid  → ~4.6 cycles
        (210, 23),   # high amplitude → ~5.3 cycles
        (215, 25),   # fast tight     → ~5.8 cycles
    ]
    _UR_JITTER = np.array([3.0, 0.5], dtype=NP_DTYPE)   # ±3 in U, ±0.5 in R

    def diverse_random_inputs(self, n_cases, rng):
        """Cycle through 4 (U, R) archetypes (with small jitter)."""
        out = []
        for i in range(int(n_cases)):
            U_b, R_b = self._ARCHETYPES_UR[i % len(self._ARCHETYPES_UR)]
            j = rng.uniform(-1.0, 1.0, size=2).astype(NP_DTYPE)
            U = float(np.clip(U_b + j[0] * self._UR_JITTER[0], *self.U_RANGE))
            R = float(np.clip(R_b + j[1] * self._UR_JITTER[1], *self.R_RANGE))
            x0 = self.initial_state_from_U(U).reshape(-1)
            th = self.theta_from_UR(U, R).reshape(-1)
            out.append(np.concatenate([x0, th]).astype(NP_DTYPE))
        return out

    def case_subtitle(self, x_input):
        r0, p0, alpha, beta, gamma, delta = x_input.tolist()
        return (f"$r_0$={r0:.3f}, $p_0$={p0:.3f}\n"
                rf"$\alpha$={alpha:.1f}, $\beta$={beta:.1f}" "\n"
                rf"$\gamma$={gamma:.1f}, $\delta$={delta:.1f}")
