"""
adalib/systems/fedbatch_bioreactor.py
Fed-batch bioreactor with Haldane kinetics.
Wraps the existing BioreactorProblem from the operator/MPC package.
"""
from __future__ import annotations
import sys
import numpy as np
import tensorflow as tf
from .base import ODESystem

from ..utils.paths import legacy_operator_root as _legacy_operator_root

_OP_PATH = str(_legacy_operator_root())
if _OP_PATH not in sys.path:
    sys.path.insert(0, _OP_PATH)


class FedBatchBioreactor(ODESystem):
    """
    4-state fed-batch bioreactor with Haldane substrate-inhibition kinetics.
    States : [Xs, Ss, Ps, Vs]  (biomass, substrate, product, volume)
    Control: inp  (feed flow rate, L/min)
    Params : Yx (yield), Sin (feed substrate conc.)
    """
    name = "fedbatch_bioreactor"
    state_names = ["Xs", "Ss", "Ps", "Vs"]
    control_names = ["inp"]
    parameter_names = ["Yx", "Sin"]

    state_bounds = {"Xs": (0.1, 3.7), "Ss": (0.0, 1.0),
                    "Ps": (0.0, 3.0), "Vs": (0.5, 5.0)}
    control_bounds = {"inp": (0.005, 0.200)}
    parameter_bounds = {"Yx": (0.3, 0.5), "Sin": (0.3, 1.5)}

    # Haldane kinetics constants
    MU_M   = 0.02
    K_M    = 0.05
    K_I    = 5.0
    V_PAR  = 0.004
    Y_P    = 1.2

    @property
    def mu_opt_substrate(self):
        import math
        return math.sqrt(self.K_M * self.K_I)

    def rhs(self, t, x, u=None, p=None):
        Xs, Ss, Ps, Vs = x
        inp = float(u) if u is not None else 0.04
        Yx  = float(p[0]) if p is not None else 0.40
        Sin = float(p[1]) if p is not None else 0.80

        Vs = max(Vs, 1e-6)
        Ss = max(Ss, 0.0)
        D  = inp / Vs
        mu = self.MU_M * Ss / (self.K_M + Ss + Ss**2 / self.K_I)

        dXs = mu * Xs - D * Xs
        dSs = -mu * Xs / Yx - self.V_PAR * Xs / self.Y_P + D * (Sin - Ss)
        dPs = self.V_PAR * Xs - D * Ps
        dVs = inp
        return np.array([dXs, dSs, dPs, dVs])

    def rhs_tf(self, var_list, i, u=None, p=None):
        Xs, Xs_t = var_list[0]
        Ss, Ss_t = var_list[1]
        Ps, Ps_t = var_list[2]
        Vs, Vs_t = var_list[3]
        dtype = Xs.dtype
        inp   = tf.cast(float(u)    if u is not None else 0.04, dtype)
        Yx    = tf.cast(float(p[0]) if p is not None else 0.40, dtype)
        Sin   = tf.cast(float(p[1]) if p is not None else 0.80, dtype)
        MU_M  = tf.cast(self.MU_M,  dtype)
        K_M   = tf.cast(self.K_M,   dtype)
        K_I   = tf.cast(self.K_I,   dtype)
        V_PAR = tf.cast(self.V_PAR, dtype)
        Y_P   = tf.cast(self.Y_P,   dtype)
        Vs_s  = tf.maximum(Vs, tf.cast(1e-6, dtype))
        Ss_s  = tf.maximum(Ss, tf.cast(0.0,  dtype))
        D     = inp / Vs_s
        mu    = MU_M * Ss_s / (K_M + Ss_s + Ss_s ** 2 / K_I)
        dXs   = mu * Xs - D * Xs
        dSs   = -mu * Xs / Yx - V_PAR * Xs / Y_P + D * (Sin - Ss_s)
        dPs   = V_PAR * Xs - D * Ps
        dVs   = inp
        rhs   = [dXs, dSs, dPs, dVs]
        return var_list[i][1] - rhs[i]
