#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/cstr_problem.py
============================================================
CSTR benchmark — 4-state Arrhenius reaction + heat balance.

States   : [C_A, C_B, T_R, T_K]    concentrations [mol/l] + temperatures [°C]
Params   : [α, β, F, Q]            Arrhenius mults + feed dilution + jacket heat

Internal time: hours (Klatt convention).
  - F: feed dilution rate [1/h]; displayed as volumetric flow [l/min]
       via F·V_R/60 in extra_plot_traces / case_subtitle.
  - Q: jacket heat duty [kJ/h] (formerly named Q_dot — same value, cleaner
       notation since "Q" itself denotes a rate in this convention).

Stiff system → reference solver = BDF (set in PROBLEM_CONFIGS["cstr"]).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from config import NP_DTYPE, TF_DTYPE
from problems.base_problem import BaseProblem

DTYPE = getattr(tf, TF_DTYPE)

# Arrhenius / heat-balance constants
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


class CSTRProblem(BaseProblem):
    name = "cstr"
    state_dim = 4
    param_dim = 4
    state_labels = ("C_A", "C_B", "T_R", "T_K")
    param_labels = ("alpha", "beta", "F", "Q")
    time_unit = "min"          # display in minutes (transient happens within ~5 min)
    time_factor = 60.0         # internal t is in hours; multiply by 60 for plotting
    ref_solver = "BDF"
    ref_rtol = 1e-8
    ref_atol = 1e-10

    # Build every one of the N_SEG segments per case (vs the default 1
    # random seg/case). The `"all"` strategy is essential for CSTR because
    # all of the Arrhenius dynamics happen in the first ~0.1h of the 0.5h
    # window — random sampling would skip the transient 80% of the time.
    # (V3-B's additional early-transient oversampling was removed: it doubled
    # wall-clock time for marginal accuracy gain. `"all"` alone already gives
    # every transient segment ~20% representation, which is sufficient.)
    segment_sampling_strategy = "all"

    ALPHA_RANGE = (0.9, 1.1)
    BETA_RANGE  = (0.9, 1.1)
    F_RANGE     = (5.0, 100.0)      # dilution rate [1/h]
    Q_RANGE     = (-8500.0, 0.0)    # heat duty [kJ/h]

    X0_LOWER   = np.array([0.1, 0.1,  50.0,  50.0], dtype=NP_DTYPE)
    X0_UPPER   = np.array([2.0, 2.0, 140.0, 140.0], dtype=NP_DTYPE)
    X0_NOMINAL = np.array([0.8, 0.5, 134.14, 130.0], dtype=NP_DTYPE)
    X0_PERTURB_FRAC = 0.05

    # Physics-informed derived features (Arrhenius CSTR):
    #   k1₀, k2₀, k3₀ at T_R₀                  (reaction rates at IC)
    #   τ_res = 1/F                              (residence time, h)
    #   ΔT₀  = T_R₀ − T_K₀                       (driving for jacket heat xfer)
    #   reaction_heat₀ at IC                     (signed energy release)
    # At T_R ≈ 100°C, k1 ≈ k2 ≈ 30, k3 ≈ 5; norms tuned for the IC ranges.
    n_derived_features = 6
    derived_mean = np.array([52.0, 52.0, 13.0, 0.04, 5.0, -460.0], dtype=NP_DTYPE)
    derived_std  = np.array([30.0, 30.0, 14.0, 0.10, 30.0,  390.0], dtype=NP_DTYPE)

    # Per-state output scaling on W: handles the ~100× magnitude gap between
    # concentrations (C_A, C_B ~ 1 mol/l) and temperatures (T_R, T_K ~ 130 °C).
    # Equal to RES_SCALE so ∂Loss/∂W is roughly state-uniform — without this,
    # C-state gradients are ~3600× larger than T-state gradients and the
    # network spends early epochs just calibrating per-state magnitudes.
    output_scale = np.array([0.5, 0.3, 30.0, 15.0], dtype=NP_DTYPE)

    # ---------------------------------------------------------------
    def sample_cases(self, n, seed):
        rng = np.random.default_rng(seed)
        alpha = rng.uniform(*self.ALPHA_RANGE, size=(n,)).astype(NP_DTYPE)
        beta  = rng.uniform(*self.BETA_RANGE,  size=(n,)).astype(NP_DTYPE)
        F     = rng.uniform(*self.F_RANGE,     size=(n,)).astype(NP_DTYPE)
        Q     = rng.uniform(*self.Q_RANGE,     size=(n,)).astype(NP_DTYPE)
        theta = np.stack([alpha, beta, F, Q], axis=-1)

        noise = rng.uniform(-1.0, 1.0, size=(n, self.state_dim)).astype(NP_DTYPE)
        span = self.X0_PERTURB_FRAC * (self.X0_UPPER - self.X0_LOWER)
        x0 = self.X0_NOMINAL[None, :] + noise * span[None, :]
        x0 = np.clip(x0, self.X0_LOWER[None, :], self.X0_UPPER[None, :]).astype(NP_DTYPE)
        return x0, theta, {}

    @staticmethod
    def _rates_np(T_R, alpha, beta):
        k1 = beta * K0_AB * np.exp(-E_A_AB / (T_R + 273.15))
        k2 = K0_BC * np.exp(-E_A_BC / (T_R + 273.15))
        k3 = K0_AD * np.exp((-alpha * E_A_AD) / (T_R + 273.15))
        return k1, k2, k3

    def rhs_np(self, t, x, theta):
        C_A, C_B, T_R, T_K = np.asarray(x, dtype=np.float64)
        alpha, beta, F, Q = np.asarray(theta, dtype=np.float64)
        k1, k2, k3 = self._rates_np(T_R, alpha, beta)
        dC_A = F * (C_A0_FEED - C_A) - k1 * C_A - k3 * C_A * C_A
        dC_B = -F * C_B + k1 * C_A - k2 * C_B
        rh = (k1 * C_A * H_R_AB + k2 * C_B * H_R_BC + k3 * C_A * C_A * H_R_AD)
        dT_R = (rh / (-RHO * CP)
                + F * (T_IN - T_R)
                + (K_W * A_R * (T_K - T_R)) / (RHO * CP * V_R))
        dT_K = (Q + K_W * A_R * (T_R - T_K)) / (M_K * CP_K)
        return np.array([dC_A, dC_B, dT_R, dT_K], dtype=np.float64)

    def rhs_tf(self, x, theta):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        theta = tf.convert_to_tensor(theta, dtype=DTYPE)
        if x.shape.rank is not None and theta.shape.rank is not None:
            if x.shape.rank == theta.shape.rank + 1:
                theta = tf.expand_dims(theta, axis=1)
        C_A, C_B, T_R, T_K = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        alpha, beta, F, Q = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]

        inv_T = 1.0 / (T_R + tf.constant(273.15, dtype=DTYPE))
        k1 = beta * K0_AB * tf.exp(-E_A_AB * inv_T)
        k2 = K0_BC * tf.exp(-E_A_BC * inv_T)
        k3 = K0_AD * tf.exp((-alpha * E_A_AD) * inv_T)

        dC_A = F * (C_A0_FEED - C_A) - k1 * C_A - k3 * tf.square(C_A)
        dC_B = -F * C_B + k1 * C_A - k2 * C_B
        rh = (k1 * C_A * H_R_AB + k2 * C_B * H_R_BC + k3 * tf.square(C_A) * H_R_AD)
        dT_R = (rh / (-RHO * CP)
                + F * (T_IN - T_R)
                + (K_W * A_R * (T_K - T_R)) / (RHO * CP * V_R))
        dT_K = (Q + K_W * A_R * (T_R - T_K)) / (M_K * CP_K)
        return tf.stack([dC_A, dC_B, dT_R, dT_K], axis=-1)

    def derived_features_tf(self, x_input):
        """6 Arrhenius CSTR features from raw [C_A, C_B, T_R, T_K, α, β, F, Q]."""
        x = tf.convert_to_tensor(x_input, dtype=DTYPE)
        eps = tf.constant(1e-6, dtype=DTYPE)
        C_A   = x[:, 0]; C_B = x[:, 1]; T_R = x[:, 2]; T_K = x[:, 3]
        alpha = x[:, 4]; beta = x[:, 5]; F   = tf.maximum(x[:, 6], eps)
        # Q = x[:, 7] — kept as raw input only

        inv_T = 1.0 / (T_R + tf.constant(273.15, dtype=DTYPE))
        k1 = beta * K0_AB * tf.exp(-E_A_AB * inv_T)
        k2 = K0_BC * tf.exp(-E_A_BC * inv_T)
        k3 = K0_AD * tf.exp((-alpha * E_A_AD) * inv_T)
        tau_res = 1.0 / F
        dT = T_R - T_K
        reaction_heat = (k1 * C_A * H_R_AB
                         + k2 * C_B * H_R_BC
                         + k3 * tf.square(C_A) * H_R_AD)
        return tf.stack([k1, k2, k3, tau_res, dT, reaction_heat], axis=-1)

    # ---------------------------------------------------------------
    def nominal_input(self):
        return np.concatenate([
            self.X0_NOMINAL,
            np.array([1.0, 1.0, 50.0, -2000.0], dtype=NP_DTYPE),
        ])

    def sweep_specs(self):
        return [
            ("$F$",       np.linspace(*self.F_RANGE,     8, dtype=NP_DTYPE), 6),
            ("$Q$",       np.linspace(*self.Q_RANGE,     8, dtype=NP_DTYPE), 7),
            (r"$\alpha$", np.linspace(*self.ALPHA_RANGE, 8, dtype=NP_DTYPE), 4),
        ]

    def random_input(self, rng):
        x0, theta, _ = self.sample_cases(1, seed=int(rng.integers(0, 2**31 - 1)))
        return self.build_input(x0[0], theta[0])

    def case_subtitle(self, x_input):
        C_A, C_B, T_R, T_K, alpha, beta, F, Q = x_input.tolist()
        F_lpm = F * V_R / 60.0   # dilution rate [1/h] → volumetric flow [l/min]
        return (f"$C_A$={C_A:.2f}, $C_B$={C_B:.2f}\n"
                f"$T_R$={T_R:.1f}, $T_K$={T_K:.1f}\n"
                rf"$\alpha$={alpha:.3f}, $\beta$={beta:.3f}" "\n"
                f"$F$={F_lpm:.2f} l/min, $Q$={Q:.0f} kJ/h")

    def state_units(self):
        return ("mol/l", "mol/l", "°C", "°C")

    def state_plot_labels(self):
        # Lower-case a, b in math mode + R, K subscripts to match the
        # reference plot style (do-mpc convention: c_a, c_b, T_R, T_K).
        return (r"$C_a$", r"$C_b$", r"$T_R$", r"$T_K$")

    # Auxiliary plot traces: ΔT (heat-transfer driving force),
    # F (feed flow), Q̇ (jacket heat). The first is derived from states,
    # the other two are case-wise constants.
    def extra_plot_traces(self):
        def delta_T(x, theta):
            return x[:, 2] - x[:, 3]

        def feed_F(x, theta):
            # Display F as volumetric flow rate [l/min] = dilution[1/h] · V_R / 60
            return np.full(x.shape[0], theta[2] * V_R / 60.0, dtype=NP_DTYPE)

        def heat_Q(x, theta):
            return np.full(x.shape[0], theta[3], dtype=NP_DTYPE)

        return [
            {"label": r"$\Delta T = T_R - T_K$", "unit": "°C",    "fn": delta_T, "kind": "derived"},
            {"label": r"$F$",                    "unit": "l/min", "fn": feed_F,  "kind": "constant"},
            {"label": r"$Q$",                    "unit": "kJ/h",  "fn": heat_Q,  "kind": "constant"},
        ]

    # 4 archetypes near nominal IC (training-friendly), varying F + Q to
    # produce visually distinct dynamics. Each row: full input vector
    # [C_A, C_B, T_R, T_K, α, β, F, Q].
    _ARCHETYPES = [
        # Nominal:        F=50, mild cooling → smooth approach to higher-T steady
        [0.80, 0.50, 134.0, 130.0, 1.00, 1.00, 50.0, -2000.0],
        # Slow-turnover:  F=20, mild cooling → low conversion, gentle T rise
        [0.75, 0.55, 132.0, 128.0, 1.00, 1.00, 20.0, -1500.0],
        # Fast-cold:      F=80, strong cooling → C_A diluted toward feed (5.1)
        [0.80, 0.50, 134.0, 130.0, 1.00, 1.05, 80.0, -6000.0],
        # Mild-heat:      F=50, weak cooling but bounded → biggest T rise
        [0.85, 0.45, 134.0, 130.0, 0.95, 1.00, 50.0, -1200.0],
    ]
    _ARCHETYPE_JITTER_SCALE = np.array(
        [0.05, 0.05, 1.0, 1.0, 0.02, 0.02, 5.0, 200.0],
        dtype=NP_DTYPE,
    )

    def diverse_random_inputs(self, n_cases, rng):
        out = []
        lo = np.concatenate([self.X0_LOWER,
                             np.array([self.ALPHA_RANGE[0], self.BETA_RANGE[0],
                                       self.F_RANGE[0],     self.Q_RANGE[0]],
                                      dtype=NP_DTYPE)])
        hi = np.concatenate([self.X0_UPPER,
                             np.array([self.ALPHA_RANGE[1], self.BETA_RANGE[1],
                                       self.F_RANGE[1],     self.Q_RANGE[1]],
                                      dtype=NP_DTYPE)])
        for i in range(int(n_cases)):
            base = np.asarray(self._ARCHETYPES[i % len(self._ARCHETYPES)], dtype=NP_DTYPE)
            j = rng.uniform(-1.0, 1.0, size=base.shape).astype(NP_DTYPE)
            x_in = np.clip(base + j * self._ARCHETYPE_JITTER_SCALE, lo, hi)
            out.append(x_in.astype(NP_DTYPE))
        return out
