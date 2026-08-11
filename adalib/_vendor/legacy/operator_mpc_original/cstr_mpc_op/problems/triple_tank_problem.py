#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/triple_tank_problem.py
============================================================
Three-tank Torricelli outflow system.

States   : [h1, h2, h3]    tank heights [cm]
Params   : [Q1, Q2]         pump inflows [cm³/s]
Dynamics:
    Q13 = a1·sn·sign(h1−h3)·√(2g|h1−h3|)
    Q32 = a3·sn·sign(h3−h2)·√(2g|h3−h2|)
    Q20 = a2·sn·√(2g·h2)
    dh1/dt = (Q1 − Q13) / A
    dh2/dt = (Q2 + Q32 − Q20) / A
    dh3/dt = (Q13 − Q32) / A
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from config import NP_DTYPE, TF_DTYPE
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)

A_TANK = 154.0
S_N    = 0.5
G_ACC  = 981.0
A1     = 0.46
A2     = 0.60
A3     = 0.45
H_MIN_FLOOR = 0.0
SQRT_EPS = 1e-8


class TripleTankProblem(BaseProblem):
    name = "triple_tank"
    state_dim = 3
    param_dim = 2
    state_labels = ("h1", "h2", "h3")
    param_labels = ("Q1", "Q2")
    time_unit = "s"

    def state_plot_labels(self):
        return (r"$h_1$", r"$h_2$", r"$h_3$")

    H1_RANGE = (1.0, 55.0)
    H2_RANGE = (1.0, 55.0)
    H3_RANGE = (1.0, 55.0)
    Q1_RANGE = (0.0, 150.0)
    Q2_RANGE = (0.0, 150.0)

    # Physics-informed derived features (Torricelli triple tank):
    #   Q13₀, Q32₀, Q20₀  (signed pipe flows at IC, cm³/s)
    #   net1₀ = Q1 − Q13₀, net2₀ = Q2 + Q32₀ − Q20₀, net3₀ = Q13₀ − Q32₀
    n_derived_features = 6
    derived_mean = np.array([0.0, 0.0, 75.0, 50.0, 0.0, 0.0], dtype=NP_DTYPE)
    derived_std  = np.array([60.0, 60.0, 40.0, 80.0, 80.0, 60.0], dtype=NP_DTYPE)

    # ---------------------------------------------------------------
    def sample_cases(self, n, seed):
        rng = np.random.default_rng(seed)
        h1 = rng.uniform(*self.H1_RANGE, size=(n,)).astype(NP_DTYPE)
        h2 = rng.uniform(*self.H2_RANGE, size=(n,)).astype(NP_DTYPE)
        h3 = rng.uniform(*self.H3_RANGE, size=(n,)).astype(NP_DTYPE)
        Q1 = rng.uniform(*self.Q1_RANGE, size=(n,)).astype(NP_DTYPE)
        Q2 = rng.uniform(*self.Q2_RANGE, size=(n,)).astype(NP_DTYPE)
        x0 = np.stack([h1, h2, h3], axis=-1)
        theta = np.stack([Q1, Q2], axis=-1)
        return x0, theta, {}

    def rhs_np(self, t, x, theta):
        h1, h2, h3 = np.asarray(x, dtype=np.float64)
        Q1, Q2 = np.asarray(theta, dtype=np.float64)
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

    def rhs_tf(self, x, theta):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        theta = tf.convert_to_tensor(theta, dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)

        h1, h2, h3 = x[..., 0], x[..., 1], x[..., 2]
        Q1, Q2 = theta[..., 0], theta[..., 1]

        h1_pos = tf.nn.relu(h1 - H_MIN_FLOOR)
        h2_pos = tf.nn.relu(h2 - H_MIN_FLOOR)
        h3_pos = tf.nn.relu(h3 - H_MIN_FLOOR)

        g  = tf.constant(G_ACC, dtype=DTYPE)
        sn = tf.constant(S_N, dtype=DTYPE)
        A_ = tf.constant(A_TANK, dtype=DTYPE)
        a1 = tf.constant(A1, dtype=DTYPE)
        a2 = tf.constant(A2, dtype=DTYPE)
        a3 = tf.constant(A3, dtype=DTYPE)
        eps = tf.constant(SQRT_EPS, dtype=DTYPE)

        dh13 = h1_pos - h3_pos
        Q13 = a1 * sn * tf.sign(dh13) * tf.sqrt(2.0 * g * tf.abs(dh13) + eps)
        dh32 = h3_pos - h2_pos
        Q32 = a3 * sn * tf.sign(dh32) * tf.sqrt(2.0 * g * tf.abs(dh32) + eps)
        Q20 = a2 * sn * tf.sqrt(2.0 * g * h2_pos + eps)

        Q1_b = tf.broadcast_to(Q1, tf.shape(h1))
        Q2_b = tf.broadcast_to(Q2, tf.shape(h2))

        dh1 = (Q1_b - Q13) / A_
        dh2 = (Q2_b + Q32 - Q20) / A_
        dh3 = (Q13 - Q32) / A_
        return tf.stack([dh1, dh2, dh3], axis=-1)

    def derived_features_tf(self, x_input):
        """6 Torricelli flow features from raw [h1, h2, h3, Q1, Q2]."""
        x = tf.convert_to_tensor(x_input, dtype=DTYPE)
        h1 = tf.nn.relu(x[:, 0] - H_MIN_FLOOR)
        h2 = tf.nn.relu(x[:, 1] - H_MIN_FLOOR)
        h3 = tf.nn.relu(x[:, 2] - H_MIN_FLOOR)
        Q1 = x[:, 3]
        Q2 = x[:, 4]

        eps = tf.constant(SQRT_EPS, dtype=DTYPE)
        g  = tf.constant(G_ACC, dtype=DTYPE)
        sn = tf.constant(S_N,   dtype=DTYPE)
        a1 = tf.constant(A1, dtype=DTYPE)
        a2 = tf.constant(A2, dtype=DTYPE)
        a3 = tf.constant(A3, dtype=DTYPE)

        dh13 = h1 - h3
        Q13 = a1 * sn * tf.sign(dh13) * tf.sqrt(2.0 * g * tf.abs(dh13) + eps)
        dh32 = h3 - h2
        Q32 = a3 * sn * tf.sign(dh32) * tf.sqrt(2.0 * g * tf.abs(dh32) + eps)
        Q20 = a2 * sn * tf.sqrt(2.0 * g * h2 + eps)
        net1 = Q1 - Q13
        net2 = Q2 + Q32 - Q20
        net3 = Q13 - Q32
        return tf.stack([Q13, Q32, Q20, net1, net2, net3], axis=-1)

    # ---------------------------------------------------------------
    def nominal_input(self):
        return np.array([20.0, 20.0, 20.0, 60.0, 60.0], dtype=NP_DTYPE)

    def sweep_specs(self):
        return [
            ("$h_{1,0}$", np.linspace(*self.H1_RANGE, 8, dtype=NP_DTYPE), 0),
            ("$Q_1$",    np.linspace(*self.Q1_RANGE, 8, dtype=NP_DTYPE), 3),
            ("$Q_2$",    np.linspace(*self.Q2_RANGE, 8, dtype=NP_DTYPE), 4),
        ]

    def case_subtitle(self, x_input):
        h1_0, h2_0, h3_0, Q1, Q2 = x_input.tolist()
        return (f"$h_{{1,0}}$={h1_0:.1f}, $h_{{2,0}}$={h2_0:.1f}, $h_{{3,0}}$={h3_0:.1f}\n"
                f"$Q_1$={Q1:.1f}, $Q_2$={Q2:.1f}")

    def state_units(self):
        return ("cm", "cm", "cm")

    # 4 archetypes spanning fill / drain / asymmetric / near-equilibrium dynamics,
    # all within training IC ranges. Format: [h1, h2, h3, Q1, Q2].
    _ARCHETYPES = [
        # Empty fill:      low IC, both pumps moderate → all 3 tanks rise
        [5.0,  5.0,  5.0,  80.0,  80.0],
        # Drain from high: high IC, weak pumps → drain dominant
        [45.0, 45.0, 45.0, 15.0,  15.0],
        # Asymmetric fill: pump 1 dominant → tank 1 outpaces tank 2
        [20.0, 20.0, 20.0, 110.0, 40.0],
        # Near equilibrium: mid IC, mid pumps → smooth approach to steady state
        [25.0, 30.0, 15.0, 60.0,  60.0],
    ]
    _ARCHETYPE_JITTER_SCALE = np.array([2.0, 2.0, 2.0, 5.0, 5.0], dtype=NP_DTYPE)

    def diverse_random_inputs(self, n_cases, rng):
        out = []
        lo = np.array([self.H1_RANGE[0], self.H2_RANGE[0], self.H3_RANGE[0],
                       self.Q1_RANGE[0], self.Q2_RANGE[0]], dtype=NP_DTYPE)
        hi = np.array([self.H1_RANGE[1], self.H2_RANGE[1], self.H3_RANGE[1],
                       self.Q1_RANGE[1], self.Q2_RANGE[1]], dtype=NP_DTYPE)
        for i in range(int(n_cases)):
            base = np.asarray(self._ARCHETYPES[i % len(self._ARCHETYPES)], dtype=NP_DTYPE)
            j = rng.uniform(-1.0, 1.0, size=base.shape).astype(NP_DTYPE)
            x_in = np.clip(base + j * self._ARCHETYPE_JITTER_SCALE, lo, hi)
            out.append(x_in.astype(NP_DTYPE))
        return out
