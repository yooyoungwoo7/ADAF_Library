"""
adalib/systems/triple_tank.py
Three-tank system with Torricelli outflow.
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

    state_bounds = {"h1": (0.0, 400.0), "h2": (0.0, 400.0), "h3": (0.0, 400.0)}
    control_bounds = {"Q1": (0.0, 200.0), "Q2": (0.0, 200.0)}

    # Tank parameters (do-mpc benchmark)
    A    = 0.3048
    a13  = 1.127e-4
    a32  = 1.127e-4
    a20  = 1.527e-4
    g    = 981.0

    def _sign(self, x):
        return 1.0 if x >= 0 else -1.0

    def rhs(self, t, x, u=None, p=None):
        h1, h2, h3 = x
        Q1 = float(u[0]) if u is not None else 68.0
        Q2 = float(u[1]) if u is not None else 68.0

        import math
        Q13 = self.a13 * self._sign(h1-h3) * math.sqrt(2*self.g*abs(h1-h3))
        Q32 = self.a32 * self._sign(h3-h2) * math.sqrt(2*self.g*abs(h3-h2))
        Q20 = self.a20 * math.sqrt(2*self.g*abs(h2))

        dh1 = (Q1/3600 - Q13) / self.A
        dh2 = (Q2/3600 + Q32 - Q20) / self.A
        dh3 = (Q13 - Q32) / self.A
        return np.array([dh1, dh2, dh3])

    def rhs_tf(self, var_list, i, u=None, p=None):
        h1, h1_t = var_list[0]
        h2, h2_t = var_list[1]
        h3, h3_t = var_list[2]
        dtype = h1.dtype
        Q1  = tf.cast(float(u[0]) if u is not None else 68.0, dtype)
        Q2  = tf.cast(float(u[1]) if u is not None else 68.0, dtype)
        a13 = tf.cast(self.a13, dtype); a32 = tf.cast(self.a32, dtype)
        a20 = tf.cast(self.a20, dtype); g   = tf.cast(self.g,   dtype)
        A   = tf.cast(self.A,   dtype)
        Q13 = a13 * tf.math.sign(h1 - h3) * tf.sqrt(tf.cast(2.0, dtype) * g * tf.abs(h1 - h3))
        Q32 = a32 * tf.math.sign(h3 - h2) * tf.sqrt(tf.cast(2.0, dtype) * g * tf.abs(h3 - h2))
        Q20 = a20 * tf.sqrt(tf.cast(2.0, dtype) * g * tf.abs(h2))
        dh1 = (Q1 / tf.cast(3600.0, dtype) - Q13) / A
        dh2 = (Q2 / tf.cast(3600.0, dtype) + Q32 - Q20) / A
        dh3 = (Q13 - Q32) / A
        rhs = [dh1, dh2, dh3]
        return var_list[i][1] - rhs[i]
