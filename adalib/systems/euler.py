"""
adalib/systems/euler.py
Euler's equation for a rigid body.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base import ODESystem


class EulerRigidBody(ODESystem):
    name = "euler_rigid_body"
    state_names = ["w1", "w2", "w3"]
    parameter_names = ["I1", "I2", "I3"]

    def __init__(self, I1=0.2, I2=0.3, I3=0.4):
        self.I1 = I1; self.I2 = I2; self.I3 = I3

    def rhs(self, t, x, u=None, p=None):
        I1 = p[0] if p is not None else self.I1
        I2 = p[1] if p is not None else self.I2
        I3 = p[2] if p is not None else self.I3
        w1, w2, w3 = x
        dw1 = ((I2 - I3) / (I2 * I3)) * w2 * w3
        dw2 = ((I3 - I1) / (I1 * I3)) * w1 * w3
        dw3 = ((I1 - I2) / (I1 * I2)) * w1 * w2
        return np.array([dw1, dw2, dw3])

    def rhs_tf(self, var_list, i, u=None, p=None):
        w1, w1_t = var_list[0]
        w2, w2_t = var_list[1]
        w3, w3_t = var_list[2]
        dtype = w1.dtype
        # Accept p=[I1, I2, I3] to enable gradient flow for inverse problems.
        # When p is None, fall back to instance attributes (forward usage).
        I1 = tf.cast(p[0] if p is not None else self.I1, dtype)
        I2 = tf.cast(p[1] if p is not None else self.I2, dtype)
        I3 = tf.cast(p[2] if p is not None else self.I3, dtype)
        c1 = (I2 - I3) / (I2 * I3)
        c2 = (I3 - I1) / (I1 * I3)
        c3 = (I1 - I2) / (I1 * I2)
        if i == 0:
            return w1_t - c1 * w2 * w3
        elif i == 1:
            return w2_t - c2 * w1 * w3
        else:
            return w3_t - c3 * w1 * w2
