#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/operator_net.py
============================================================
Vanilla MLP operator network shared by all problems.

Optionally augments the raw input with physics-informed derived features
provided by the Problem (`problem.derived_features_tf(x)`,
`problem.n_derived_features`, `problem.derived_mean / derived_std`).
If the problem does not provide derived features, the network behaves as a
plain MLP on the raw input.

Output shape: (B, state_dim, N_p)  — W panel weights only (never Legendre
amplitudes directly).
"""

import numpy as np
import tensorflow as tf

from config import TF_DTYPE, INPUT_DIM, STATE_DIM, LPA_N_P

DTYPE = getattr(tf, TF_DTYPE)


class OperatorNet(tf.keras.Model):
    """MLP: (B, input_dim [+ n_derived]) → (B, state_dim, N_p) panel weights.

    Args:
        input_dim, state_dim, N_p: ODE shape.
        hidden, n_layers, dropout: MLP shape.
        x_mean, x_std: raw-input normalisation (numpy, shape (input_dim,)).
        derived_fn: optional callable (x: (B, input_dim) → (B, n_derived))
                    that returns physics-informed features. If None, behaves
                    as a vanilla MLP.
        n_derived:  must equal derived_fn output width.
        derived_mean, derived_std: derived-feature normalisation
                    (numpy, shape (n_derived,)). Defaults to zeros / ones.
    """

    def __init__(self, input_dim=INPUT_DIM, state_dim=STATE_DIM, N_p=LPA_N_P,
                 hidden=256, n_layers=3, x_mean=None, x_std=None, dropout=0.0,
                 derived_fn=None, n_derived=0, derived_mean=None, derived_std=None,
                 output_scale=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.N_p = int(N_p)
        self.out_dim = self.state_dim * self.N_p

        # raw-input normalisation
        if x_mean is None:
            x_mean = np.zeros((self.input_dim,), dtype=np.float32)
        if x_std is None:
            x_std = np.ones((self.input_dim,), dtype=np.float32)
        x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        x_std  = np.maximum(np.asarray(x_std, dtype=np.float32).reshape(1, -1), 1e-6)
        self.x_mean = tf.constant(x_mean, dtype=DTYPE)
        self.x_std  = tf.constant(x_std,  dtype=DTYPE)

        # Per-state output scaling on W. The basis is linear in W, so scaling
        # W by `output_scale[i]` for state i scales x_delta, ẋ, ẍ uniformly
        # for that state — without breaking the hard IC (x(0)=x0, ẋ(0)=ẋ0).
        # Setting `output_scale ≈ RES_SCALE` cancels the per-state weighting
        # of the residual loss, so ∂Loss/∂W is roughly state-uniform and the
        # optimiser doesn't waste epochs calibrating per-state magnitudes.
        if output_scale is None:
            self.output_scale = None
        else:
            os_arr = np.asarray(output_scale, dtype=np.float32).reshape(self.state_dim, 1)
            self.output_scale = tf.constant(os_arr, dtype=DTYPE)

        # derived-feature augmentation
        self.derived_fn = derived_fn
        self.n_derived  = int(n_derived) if derived_fn is not None else 0
        if self.n_derived > 0:
            if derived_mean is None:
                derived_mean = np.zeros((self.n_derived,), dtype=np.float32)
            if derived_std is None:
                derived_std = np.ones((self.n_derived,), dtype=np.float32)
            dm = np.asarray(derived_mean, dtype=np.float32).reshape(1, -1)
            ds = np.maximum(np.asarray(derived_std, dtype=np.float32).reshape(1, -1), 1e-6)
            assert dm.shape[1] == self.n_derived, "derived_mean shape mismatch"
            assert ds.shape[1] == self.n_derived, "derived_std shape mismatch"
            self.derived_mean = tf.constant(dm, dtype=DTYPE)
            self.derived_std  = tf.constant(ds, dtype=DTYPE)

        total_in = self.input_dim + self.n_derived
        layers = [tf.keras.layers.Input(shape=(total_in,))]
        for _ in range(int(n_layers)):
            layers.append(tf.keras.layers.Dense(hidden, activation="swish"))
            if dropout > 0:
                layers.append(tf.keras.layers.Dropout(float(dropout)))
        # Zero-initialise the projection to W. The basis then starts from a
        # physically-valid baseline at every t∈[0, T_seg]:
        #   • LPA   →  x(t) = x0           (identity)
        #   • ADA-F →  x(t) = x0 + ẋ0·t    (affine free-part only)
        # This avoids huge initial residuals on stiff problems (e.g. CSTR
        # Arrhenius exp() overflow) where s₁ ≈ 100 would otherwise amplify
        # random initial weights into out-of-physical-range states.
        layers.append(tf.keras.layers.Dense(
            self.out_dim, activation=None,
            kernel_initializer="zeros", bias_initializer="zeros",
        ))
        self.net = tf.keras.Sequential(layers)

    def normalize(self, x):
        return (x - self.x_mean) / self.x_std

    def call(self, x, training=False):
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        raw_norm = self.normalize(x)
        if self.n_derived > 0:
            derived = self.derived_fn(x)                                 # (B, n_derived)
            derived_norm = (derived - self.derived_mean) / self.derived_std
            z = tf.concat([raw_norm, derived_norm], axis=-1)
        else:
            z = raw_norm
        y = self.net(z, training=training)
        y = tf.reshape(y, (-1, self.state_dim, self.N_p))
        if self.output_scale is not None:
            y = y * self.output_scale[None, :, :]   # broadcast: (1, state_dim, 1)
        return y
