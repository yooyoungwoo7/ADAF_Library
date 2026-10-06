"""
adalib/systems/triple_tank.py
Three-tank system with Torricelli outflow.

Uses the same governing equations and constants as the legacy
TripleTankProblem (operator/MPC package) that actually generated the
paper's training data and Table 15 (tbl:op_hparams) / Table 4
(tbl:op_tank) results — see adalib/_vendor/legacy/operator_mpc_original/
cstr_mpc_op/problems/triple_tank_problem.py. This class wraps that
problem's rhs_np/rhs_tf exactly (cm, cm^3/s, seconds; no do-mpc-style
m/m^3/h unit conversion).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base import ODESystem


class TripleTank(ODESystem):
    name = "triple_tank"
    state_names = ["h1", "h2", "h3"]
    control_names = ["Q1", "Q2"]
    parameter_names = []

    state_bounds = {"h1": (1.0, 55.0), "h2": (1.0, 55.0), "h3": (1.0, 55.0)}
    control_bounds = {"Q1": (0.0, 150.0), "Q2": (0.0, 150.0)}

    # Tank parameters (match TripleTankProblem exactly: cm, cm^3/s)
    A_TANK = 154.0
    S_N    = 0.5
    G_ACC  = 981.0
    A1     = 0.46
    A2     = 0.60
    A3     = 0.45
    H_MIN_FLOOR = 0.0
    SQRT_EPS = 1e-8

    def _sign(self, x):
        return 1.0 if x >= 0 else (-1.0 if x < 0 else 0.0)

    def rhs(self, t, x, u=None, p=None):
        h1, h2, h3 = x
        Q1 = float(u[0]) if u is not None else 60.0
        Q2 = float(u[1]) if u is not None else 60.0

        h1p = max(h1 - self.H_MIN_FLOOR, 0.0)
        h2p = max(h2 - self.H_MIN_FLOOR, 0.0)
        h3p = max(h3 - self.H_MIN_FLOOR, 0.0)

        dh13 = h1p - h3p
        dh32 = h3p - h2p
        Q13 = self.A1 * self.S_N * self._sign(dh13) * np.sqrt(
            2.0 * self.G_ACC * abs(dh13) + self.SQRT_EPS)
        Q32 = self.A3 * self.S_N * self._sign(dh32) * np.sqrt(
            2.0 * self.G_ACC * abs(dh32) + self.SQRT_EPS)
        Q20 = self.A2 * self.S_N * np.sqrt(2.0 * self.G_ACC * h2p + self.SQRT_EPS)

        dh1 = (Q1 - Q13) / self.A_TANK
        dh2 = (Q2 + Q32 - Q20) / self.A_TANK
        dh3 = (Q13 - Q32) / self.A_TANK
        return np.array([dh1, dh2, dh3])

    def rhs_tf(self, var_list, i, u=None, p=None):
        h1, h1_t = var_list[0]
        h2, h2_t = var_list[1]
        h3, h3_t = var_list[2]
        dtype = h1.dtype
        Q1 = tf.cast(float(u[0]) if u is not None else 60.0, dtype)
        Q2 = tf.cast(float(u[1]) if u is not None else 60.0, dtype)

        a1  = tf.cast(self.A1, dtype); a2  = tf.cast(self.A2, dtype)
        a3  = tf.cast(self.A3, dtype); sn  = tf.cast(self.S_N, dtype)
        g   = tf.cast(self.G_ACC, dtype); A_ = tf.cast(self.A_TANK, dtype)
        eps = tf.cast(self.SQRT_EPS, dtype)

        h1p = tf.nn.relu(h1 - self.H_MIN_FLOOR)
        h2p = tf.nn.relu(h2 - self.H_MIN_FLOOR)
        h3p = tf.nn.relu(h3 - self.H_MIN_FLOOR)

        dh13 = h1p - h3p
        Q13 = a1 * sn * tf.sign(dh13) * tf.sqrt(2.0 * g * tf.abs(dh13) + eps)
        dh32 = h3p - h2p
        Q32 = a3 * sn * tf.sign(dh32) * tf.sqrt(2.0 * g * tf.abs(dh32) + eps)
        Q20 = a2 * sn * tf.sqrt(2.0 * g * h2p + eps)

        dh1 = (Q1 - Q13) / A_
        dh2 = (Q2 + Q32 - Q20) / A_
        dh3 = (Q13 - Q32) / A_
        rhs = [dh1, dh2, dh3]
        return var_list[i][1] - rhs[i]
