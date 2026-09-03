"""
adalib/systems/cstr.py
CSTR with Arrhenius kinetics and heat balance.
Uses the same governing equations and constants as the legacy
CstrMpcProblem (operator/MPC package); matches that problem exactly at
its fixed operating point (alpha=beta=1, F=50), and generalizes
alpha/beta/F to user-supplied parameters, which CstrMpcProblem does
not expose.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base import ODESystem


class CSTR(ODESystem):
    """
    4-state CSTR with Arrhenius kinetics (A->B->C).
    States : [C_A, C_B, T_R, T_K]
    Control: Q_dot (jacket heat flow, kJ/h)
    Params : alpha, beta, F (dilution rate)
    """
    name = "cstr"
    state_names = ["C_A", "C_B", "T_R", "T_K"]
    control_names = ["Q_dot"]
    parameter_names = ["alpha", "beta", "F"]

    state_bounds = {"C_A": (0.1, 2.1), "C_B": (0.1, 2.1),
                    "T_R": (50.0, 180.0), "T_K": (50.0, 180.0)}
    control_bounds = {"Q_dot": (-8500.0, 0.0)}
    parameter_bounds = {"alpha": (1.0, 1.0), "beta": (1.0, 1.0), "F": (50.0, 50.0)}

    # CSTR constants (from do-mpc benchmark, matching
    # adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/
    # cstr_mpc_problem.py exactly — this class wraps that problem)
    K0_AB = 1.287e12
    K0_BC = 1.287e12
    K0_AD = 9.043e9
    EA_R_AB = 9758.3
    EA_R_BC = 9758.3
    EA_R_AD = 8560.0
    H_AB = 4.2
    H_BC = -11.0
    H_AD = -41.85
    RHO   = 0.9342
    CP    = 3.01
    CPK   = 2.0
    A     = 0.215
    V_R   = 10.01
    MK    = 5.0
    T_IN  = 130.0
    C_A0_FEED = (5.7 + 4.5) / 2.0
    K_W   = 4032.0

    def rhs(self, t, x, u=None, p=None):
        C_A, C_B, T_R, T_K = x
        Q_dot = float(u) if u is not None else -5000.0
        alpha  = float(p[0]) if p is not None else 1.0
        beta   = float(p[1]) if p is not None else 1.0
        F      = float(p[2]) if p is not None else 50.0  # dilution rate, 1/h

        import math
        Tc = T_R + 273.15
        k1 = beta  * self.K0_AB * math.exp(-self.EA_R_AB / Tc)
        k2 =         self.K0_BC * math.exp(-self.EA_R_BC / Tc)
        k3 =         self.K0_AD * math.exp(-alpha * self.EA_R_AD / Tc)

        dC_A = F * (self.C_A0_FEED - C_A) - k1 * C_A - k3 * C_A**2
        dC_B = -F * C_B + k1 * C_A - k2 * C_B
        rh = k1 * C_A * self.H_AB + k2 * C_B * self.H_BC + k3 * C_A**2 * self.H_AD
        dT_R = (rh / (-self.RHO * self.CP)
                + F * (self.T_IN - T_R)
                + (self.K_W * self.A * (T_K - T_R)) / (self.RHO * self.CP * self.V_R))
        dT_K = (Q_dot + self.K_W * self.A * (T_R - T_K)) / (self.MK * self.CPK)
        return np.array([dC_A, dC_B, dT_R, dT_K])

    def rhs_tf(self, var_list, i, u=None, p=None):
        C_A, C_A_t = var_list[0]
        C_B, C_B_t = var_list[1]
        T_R, T_R_t = var_list[2]
        T_K, T_K_t = var_list[3]
        dtype  = C_A.dtype
        Q_dot  = tf.cast(float(u)    if u is not None else -5000.0, dtype)
        alpha  = tf.cast(float(p[0]) if p is not None else 1.0,     dtype)
        beta   = tf.cast(float(p[1]) if p is not None else 1.0,     dtype)
        F      = tf.cast(float(p[2]) if p is not None else 50.0,    dtype)
        K0_AB  = tf.cast(self.K0_AB,  dtype); K0_BC = tf.cast(self.K0_BC, dtype)
        K0_AD  = tf.cast(self.K0_AD,  dtype)
        EA_AB  = tf.cast(self.EA_R_AB, dtype); EA_BC = tf.cast(self.EA_R_BC, dtype)
        EA_AD  = tf.cast(self.EA_R_AD, dtype)
        H_AB   = tf.cast(self.H_AB,  dtype); H_BC = tf.cast(self.H_BC, dtype)
        H_AD   = tf.cast(self.H_AD,  dtype)
        RHO    = tf.cast(self.RHO,   dtype); CP  = tf.cast(self.CP,  dtype)
        CPK    = tf.cast(self.CPK,   dtype); A   = tf.cast(self.A,   dtype)
        V_R    = tf.cast(self.V_R,   dtype); MK  = tf.cast(self.MK,  dtype)
        T_IN   = tf.cast(self.T_IN,  dtype); K_W = tf.cast(self.K_W, dtype)
        C_A0_FEED = tf.cast(self.C_A0_FEED, dtype)
        Tc = T_R + tf.constant(273.15, dtype=dtype)
        k1 = beta  * K0_AB * tf.exp(-EA_AB / Tc)
        k2 =         K0_BC * tf.exp(-EA_BC / Tc)
        k3 =         K0_AD * tf.exp(-alpha * EA_AD / Tc)
        dC_A = F * (C_A0_FEED - C_A) - k1 * C_A - k3 * C_A ** 2
        dC_B = -F * C_B + k1 * C_A - k2 * C_B
        rh = k1 * C_A * H_AB + k2 * C_B * H_BC + k3 * C_A ** 2 * H_AD
        dT_R = (rh / (-RHO * CP)
                + F * (T_IN - T_R)
                + (K_W * A * (T_K - T_R)) / (RHO * CP * V_R))
        dT_K = (Q_dot + K_W * A * (T_R - T_K)) / (MK * CPK)
        rhs  = [dC_A, dC_B, dT_R, dT_K]
        return var_list[i][1] - rhs[i]
