"""
adalib/systems/lotka_volterra.py
Lotka-Volterra predator-prey system.

Two variants:
  LotkaVolterra     — params: [alpha, beta, gamma, delta]
  LotkaVolterraUR   — params: [U, R]  (paper scaling)

LotkaVolterraUR uses the paper formulation:
  dr/dt = (R/U)(2Ur - 0.04U²rp) = 2R·r  - 0.04·R·U·r·p
  dp/dt = (R/U)(0.02U²rp - 1.06Up) = 0.02·R·U·r·p - 1.06·R·p
with fixed coefficients (2, 0.04, 0.02, 1.06) and free (U, R).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base import ODESystem


class LotkaVolterra(ODESystem):
    name = "lotka_volterra"
    state_names = ["U", "R"]
    parameter_names = ["alpha", "beta", "gamma", "delta"]

    def __init__(self, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

    def rhs(self, t, x, u=None, p=None):
        U, R = x
        a = p[0] if p is not None else self.alpha
        b = p[1] if p is not None else self.beta
        g = p[2] if p is not None else self.gamma
        d = p[3] if p is not None else self.delta
        dU = a * U - b * U * R
        dR = -g * R + d * U * R
        return np.array([dU, dR])

    def rhs_tf(self, var_list, i, u=None, p=None):
        U_s, U_t = var_list[0]
        R_s, R_t = var_list[1]
        dtype = U_s.dtype
        a = tf.cast(p[0] if p is not None else self.alpha, dtype)
        b = tf.cast(p[1] if p is not None else self.beta,  dtype)
        g = tf.cast(p[2] if p is not None else self.gamma, dtype)
        d = tf.cast(p[3] if p is not None else self.delta, dtype)
        if i == 0:
            return U_t - (a * U_s - b * U_s * R_s)
        else:
            return R_t - (-g * R_s + d * U_s * R_s)


class LotkaVolterraUR(ODESystem):
    """Lotka-Volterra in paper (U, R) scaling.

    dr/dt = 2R·r  - 0.04·R·U·r·p
    dp/dt = 0.02·R·U·r·p - 1.06·R·p

    Parameters: p = [U, R]
      U  — prey scale  (true ≈ 200)
      R  — rate scale  (true ≈ 20)
    """
    name = "lotka_volterra_ur"
    state_names = ["r (prey)", "p (predator)"]
    parameter_names = ["U", "R"]

    def __init__(self, U=200.0, R=20.0):
        self.U = U
        self.R = R

    def rhs(self, t, x, u=None, p=None):
        r, pred = x
        U_ = p[0] if p is not None else self.U
        R_ = p[1] if p is not None else self.R
        dr   =  2.0   * R_ * r      - 0.04 * R_ * U_ * r * pred
        dpred = 0.02  * R_ * U_ * r * pred - 1.06 * R_ * pred
        return np.array([dr, dpred])

    def rhs_tf(self, var_list, i, u=None, p=None):
        r_s,    r_t    = var_list[0]
        pred_s, pred_t = var_list[1]
        dtype = r_s.dtype
        U_ = tf.cast(p[0] if p is not None else self.U, dtype)
        R_ = tf.cast(p[1] if p is not None else self.R, dtype)
        if i == 0:
            f = 2.0 * R_ * r_s - 0.04 * R_ * U_ * r_s * pred_s
            return r_t - f
        else:
            f = 0.02 * R_ * U_ * r_s * pred_s - 1.06 * R_ * pred_s
            return pred_t - f
