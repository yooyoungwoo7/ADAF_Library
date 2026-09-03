#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/bioreactor_problem.py
============================================================
Fed-batch bioreactor with Haldane substrate-inhibition kinetics.

States   : [Xs, Ss, Ps, Vs]   biomass, substrate, product, volume
Params   : [Yx, Sin, inp]     yield, feed conc., feed rate
Dynamics:
    μ      = MU_M · Ss / (K_M + Ss + Ss²/K_I)
    D      = inp / Vs
    dXs/dt = μ Xs − D Xs
    dSs/dt = −μ Xs/Yx − V_PAR Xs/Y_P + D (Sin − Ss)
    dPs/dt = V_PAR Xs − D Ps
    dVs/dt = inp
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from config import NP_DTYPE, TF_DTYPE, DT_SEG
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)

MU_M    = 0.02
K_M     = 0.05
K_I     = 5.0
V_PAR   = 0.004
Y_P     = 1.2
VS_FLOOR = 1e-6
SS_FLOOR = 0.0


class BioreactorProblem(BaseProblem):
    name = "bioreactor"
    state_dim = 4
    param_dim = 3
    state_labels = ("Xs", "Ss", "Ps", "Vs")
    param_labels = ("Yx", "Sin", "inp")
    time_unit = "min"

    def state_plot_labels(self):
        return (r"$X_s$", r"$S_s$", r"$P_s$", r"$V_s$")

    # Physics-informed derived features (Haldane bioreactor):
    #   μ₀ = MU_M·Ss₀ / (K_M + Ss₀ + Ss₀²/K_I)   (specific growth at IC)
    #   D₀ = inp / Vs₀                             (dilution rate)
    #   growth_X₀ = (μ₀ − D₀)·Xs₀                  (instant biomass derivative)
    #   growth_S₀ = −μ₀ Xs₀/Yx − V_PAR Xs₀/Y_P + D₀(Sin − Ss₀)
    #   inhib₀ = Ss₀ / K_I                          (Haldane inhibition severity)
    #   Vs_end_est = Vs₀ + inp · DT_SEG             (volume at segment end)
    n_derived_features = 6
    derived_mean = np.array([0.010, 0.060, -0.080, 11.0, 0.50, 2.90], dtype=NP_DTYPE)
    derived_std  = np.array([0.005, 0.060,  0.120, 12.0, 0.30, 1.35], dtype=NP_DTYPE)

    # Per-state output scaling on W: bioreactor RES_SCALE = (0.033, 3.4, 0.025,
    # 0.1) spans ~136× (Ss is much faster than the others). Without this, the
    # ∂Loss/∂W ratio between Ps and Ss states is ~18,000×, which forces the
    # network to learn dramatically different W magnitudes per state. Setting
    # output_scale = RES_SCALE cancels this imbalance.
    output_scale = np.array([0.033, 3.4, 0.025, 0.1], dtype=NP_DTYPE)

    YX_RANGE  = (0.3, 0.5)
    SIN_RANGE = (180.0, 220.0)
    INP_RANGE = (0.005, 0.2)
    XS0_RANGE = (0.1, 3.7)
    SS0_RANGE = (0.0, 5.0)
    PS0_RANGE = (0.0, 3.0)
    VS0_RANGE = (0.5, 5.0)

    # ---------------------------------------------------------------
    def sample_cases(self, n, seed):
        rng = np.random.default_rng(seed)
        Xs0 = rng.uniform(*self.XS0_RANGE, size=(n,)).astype(NP_DTYPE)
        Ss0 = rng.uniform(*self.SS0_RANGE, size=(n,)).astype(NP_DTYPE)
        Ps0 = rng.uniform(*self.PS0_RANGE, size=(n,)).astype(NP_DTYPE)
        Vs0 = rng.uniform(*self.VS0_RANGE, size=(n,)).astype(NP_DTYPE)
        Yx  = rng.uniform(*self.YX_RANGE,  size=(n,)).astype(NP_DTYPE)
        Sin = rng.uniform(*self.SIN_RANGE, size=(n,)).astype(NP_DTYPE)
        inp = rng.uniform(*self.INP_RANGE, size=(n,)).astype(NP_DTYPE)
        x0 = np.stack([Xs0, Ss0, Ps0, Vs0], axis=-1)
        theta = np.stack([Yx, Sin, inp], axis=-1)
        return x0, theta, {}

    def rhs_np(self, t, x, theta):
        Xs, Ss, Ps, Vs = np.asarray(x, dtype=np.float64)
        Yx, Sin, inp = np.asarray(theta, dtype=np.float64)
        Vs_safe = max(Vs, VS_FLOOR)
        Ss_safe = max(Ss, SS_FLOOR)
        D = inp / Vs_safe
        mu = MU_M * Ss_safe / (K_M + Ss_safe + Ss_safe * Ss_safe / K_I)
        return np.array([
            mu * Xs - D * Xs,
            -mu * Xs / Yx - V_PAR * Xs / Y_P + D * (Sin - Ss),
            V_PAR * Xs - D * Ps,
            inp,
        ], dtype=np.float64)

    def rhs_tf(self, x, theta):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        theta = tf.convert_to_tensor(theta, dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)
        Xs, Ss, Ps, Vs = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        Yx, Sin, inp = theta[..., 0], theta[..., 1], theta[..., 2]

        Ss_safe = tf.nn.relu(Ss)
        Vs_safe = tf.maximum(Vs, tf.constant(VS_FLOOR, dtype=DTYPE))
        D = inp / Vs_safe
        mu = MU_M * Ss_safe / (K_M + Ss_safe + tf.square(Ss_safe) / K_I)

        dXs = mu * Xs - D * Xs
        dSs = -mu * Xs / Yx - V_PAR * Xs / Y_P + D * (Sin - Ss)
        dPs = V_PAR * Xs - D * Ps
        dVs = tf.broadcast_to(inp, tf.shape(Xs))
        return tf.stack([dXs, dSs, dPs, dVs], axis=-1)

    def derived_features_tf(self, x_input):
        """6 Haldane physics features from raw [Xs0, Ss0, Ps0, Vs0, Yx, Sin, inp]."""
        x = tf.convert_to_tensor(x_input, dtype=DTYPE)
        eps = tf.constant(1e-6, dtype=DTYPE)
        Xs0 = x[:, 0]
        Ss0 = tf.maximum(x[:, 1], tf.constant(SS_FLOOR, dtype=DTYPE))
        Vs0 = tf.maximum(x[:, 3], tf.constant(VS_FLOOR, dtype=DTYPE))
        Yx  = tf.maximum(x[:, 4], eps)
        Sin = x[:, 5]
        inp = x[:, 6]

        mu0 = MU_M * Ss0 / (K_M + Ss0 + tf.square(Ss0) / K_I)
        D0  = inp / Vs0
        growth_X = (mu0 - D0) * Xs0
        growth_S = -mu0 * Xs0 / Yx - V_PAR * Xs0 / Y_P + D0 * (Sin - Ss0)
        inhib = Ss0 / K_I
        Vs_end = Vs0 + inp * tf.constant(float(DT_SEG), dtype=DTYPE)
        return tf.stack([mu0, D0, growth_X, growth_S, inhib, Vs_end], axis=-1)

    # ---------------------------------------------------------------
    def nominal_input(self):
        return np.array([1.0, 0.5, 0.0, 2.0, 0.4, 200.0, 0.05], dtype=NP_DTYPE)

    def sweep_specs(self):
        return [
            ("$Y_x$",   np.linspace(*self.YX_RANGE,  8, dtype=NP_DTYPE), 4),
            ("$X_{s0}$", np.linspace(*self.XS0_RANGE, 8, dtype=NP_DTYPE), 0),
            ("inp",     np.linspace(*self.INP_RANGE, 8, dtype=NP_DTYPE), 6),
        ]

    def case_subtitle(self, x_input):
        Xs0, Ss0, Ps0, Vs0, Yx, Sin, inp = x_input.tolist()
        return (f"$X_{{s0}}$={Xs0:.2f}, $S_{{s0}}$={Ss0:.2f}\n"
                f"$Y_x$={Yx:.3f}, $S_{{in}}$={Sin:.1f}\n"
                f"inp={inp:.3f}, $V_{{s0}}$={Vs0:.2f}")

    # ---------------------------------------------------------------
    # 4 dynamical archetypes for plot_random_cases — chosen to span the
    # fed-batch regimes (growth / substrate-limited / Haldane-inhibited /
    # washout). Calling with n_cases > 4 cycles and adds rng jitter.
    # ---------------------------------------------------------------
    # 4 archetypes chosen to (1) span visually distinct trajectories AND
    # (2) sit within the training distribution so the operator predicts well.
    # All parameters kept clear of the IC/θ range boundaries.
    _ARCHETYPES = [
        # Visible growth:    low Xs₀, optimal Ss₀, slow feed → Xs rises ~1.3×
        # (no boundary params; previous version had Ss₀=0, inp=0.005, Yx=0.50
        # which were all at the edges of training distribution)
        dict(Xs0=0.4,  Ss0=0.5, Ps0=0.2, Vs0=4.0, Yx=0.45, Sin=195.0, inp=0.010, tag="growth"),
        # High-biomass:      mid-high Xs₀, gentle decline + product accumulation
        dict(Xs0=3.0,  Ss0=1.0, Ps0=0.5, Vs0=2.0, Yx=0.45, Sin=200.0, inp=0.015, tag="high-prod"),
        # Dramatic decline:  Xs crashes from 2.5 → 0.4 (clear washout shape)
        dict(Xs0=2.5,  Ss0=2.5, Ps0=0.5, Vs0=1.5, Yx=0.35, Sin=200.0, inp=0.080, tag="washout"),
        # Slow inhibited:    moderate Ss₀ + moderate feed → mid-tier decline
        dict(Xs0=1.0,  Ss0=3.0, Ps0=0.3, Vs0=2.5, Yx=0.40, Sin=200.0, inp=0.040, tag="inhib"),
    ]
    _ARCHETYPE_JITTER_SCALE = np.array([0.15, 0.10, 0.10, 0.20, 0.02, 4.0, 0.002], dtype=NP_DTYPE)

    def diverse_random_inputs(self, n_cases, rng):
        """Return n_cases archetype-cycled fed-batch inputs (jitter on top)."""
        out = []
        for i in range(int(n_cases)):
            arch = self._ARCHETYPES[i % len(self._ARCHETYPES)]
            base = np.array(
                [arch["Xs0"], arch["Ss0"], arch["Ps0"], arch["Vs0"],
                 arch["Yx"],  arch["Sin"], arch["inp"]],
                dtype=NP_DTYPE,
            )
            jitter = rng.uniform(-1.0, 1.0, size=base.shape).astype(NP_DTYPE)
            x_in = base + jitter * self._ARCHETYPE_JITTER_SCALE
            # clip to training ranges so we don't extrapolate
            lo = np.array([self.XS0_RANGE[0], self.SS0_RANGE[0], self.PS0_RANGE[0],
                           self.VS0_RANGE[0], self.YX_RANGE[0],  self.SIN_RANGE[0],
                           self.INP_RANGE[0]], dtype=NP_DTYPE)
            hi = np.array([self.XS0_RANGE[1], self.SS0_RANGE[1], self.PS0_RANGE[1],
                           self.VS0_RANGE[1], self.YX_RANGE[1],  self.SIN_RANGE[1],
                           self.INP_RANGE[1]], dtype=NP_DTYPE)
            out.append(np.clip(x_in, lo, hi).astype(NP_DTYPE))
        return out
