"""
adalib/systems/pendulum.py
Damped nonlinear pendulum system.

  d(theta)/dt = omega
  d(omega)/dt = -gamma * omega - g_over_l * sin(theta)

Parameters: p = [gamma, g_over_l]
  gamma    — viscous damping coefficient [1/s]
  g_over_l — gravitational acceleration / rod length [rad/s^2]
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base import ODESystem


class DampedPendulum(ODESystem):
    name = "damped_pendulum"
    state_names = ["theta", "omega"]
    parameter_names = ["gamma", "g_over_l"]

    def __init__(self, gamma: float = 0.30, g_over_l: float = 9.81):
        self.gamma    = gamma
        self.g_over_l = g_over_l

    def rhs(self, t, x, u=None, p=None):
        gamma    = p[0] if p is not None else self.gamma
        g_over_l = p[1] if p is not None else self.g_over_l
        theta, omega = x
        dtheta = omega
        domega = -gamma * omega - g_over_l * np.sin(theta)
        return np.array([dtheta, domega])

    def rhs_tf(self, var_list, i, u=None, p=None):
        theta, theta_t = var_list[0]
        omega, omega_t = var_list[1]
        dtype = theta.dtype
        gamma    = tf.cast(p[0] if p is not None else self.gamma,    dtype)
        g_over_l = tf.cast(p[1] if p is not None else self.g_over_l, dtype)
        if i == 0:
            return theta_t - omega
        else:
            return omega_t - (-gamma * omega - g_over_l * tf.sin(theta))
