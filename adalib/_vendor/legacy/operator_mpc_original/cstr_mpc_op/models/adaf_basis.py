#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/adaf_basis.py
============================================================
Adaptive Discrete Approximation – Fourier (ADA-F) basis with HARD enforcement
of both x(0)=x0 and ẋ(0)=ẋ0=f(x0, θ).

Network outputs panel weights W ∈ ℝ^{state_dim × N_p} on N_p collocation
points. Fourier coefficients (a₀, aₙ, bₙ) are obtained via the cosine/sine
"DCT-like" projection:

    a₀ = mean_p(W_p)
    aₙ = (4/π)·1/(2n−1) · Σ_p sin(nπ·(2p−1)/(2N_p)) · W_p     n = 1..N_m
    bₙ = (4/π)·1/(2n−1) · Σ_p cos(nπ·(2p−1)/(2N_p)) · W_p

Forward construction (per state, with s₁ = 2/T_seg, ξ = s₁·t − 1):

    ẍ_local(ξ) = ½a₀ + ½ Σₙ [aₙ cos(nπ(ξ+1)) + bₙ sin(nπ(ξ+1))]

    xdot_free(ξ) = ∫_{-1}^{ξ} ẍ_local dξ'    [hard-zeros at ξ=-1]
                = ½ a₀ (ξ+1) + ½ Σₙ [factor·aₙ·Sₙ − factor·bₙ·(Cₙ−1)]

    x_free(ξ)  = ∫_{-1}^{ξ} xdot_free dξ'    [hard-zeros at ξ=-1]
                = ¼ a₀ (ξ+1)² + ½ Σₙ [factor²·aₙ·(1−Cₙ) − factor²·bₙ·Sₙ
                                       + factor·bₙ·(ξ+1)]

    where factor = 1/(nπ), Cₙ = cos(nπ(ξ+1)), Sₙ = sin(nπ(ξ+1)).

Then by chain rule (d/dt = s₁·d/dξ):
    x(t)  = x0 + ẋ0·t + x_free(ξ(t))
    ẋ(t) = ẋ0 + s₁ · xdot_free(ξ(t))
    ẍ(t) = s₁² · ẍ_local(ξ(t))

Since x_free(-1)=0 and xdot_free(-1)=0, x(0)=x0 and ẋ(0)=ẋ0 are exact.

Output dict matches BatchLPABasis: {"x", "xdot", "xddot", "xdddot", "x_end",
"t_local"}.
"""

import numpy as np
import tensorflow as tf

from config import TF_DTYPE, DT_SEG, NT_SEG, ADAF_N_M, ADAF_N_P


class BatchADAFBasis(tf.Module):
    """Fourier basis built around the affine free part x0 + ẋ0·t. ICs are
    encouraged through training (residual physics drives the deviation to
    zero) rather than enforced symbolically."""

    def __init__(self, T_seg=DT_SEG, Nt=NT_SEG, N_m=ADAF_N_M, N_p=ADAF_N_P,
                 dtype=None, name=None):
        super().__init__(name=name)
        if dtype is None:
            dtype = getattr(tf, TF_DTYPE)
        self.dtype = dtype
        self.T_seg = float(T_seg)
        self.Nt = int(Nt)
        self.N_m = int(N_m)
        self.N_p = int(N_p)
        self.coef_dim = self.N_p

        self.t_local_np = np.linspace(0.0, self.T_seg, self.Nt, dtype=np.float32)
        self.t_local = tf.constant(self.t_local_np, dtype=dtype)

        xi_np = 2.0 * (self.t_local_np / self.T_seg) - 1.0
        self.xi_np = xi_np.astype(np.float32)
        self.xi = tf.constant(self.xi_np, dtype=dtype)
        self.time_scale = tf.constant(2.0 / self.T_seg, dtype=dtype)

        m_idx = np.arange(self.N_m, dtype=np.float32)
        p_idx = np.arange(self.N_p, dtype=np.float32)

        Mb = np.cos(np.pi * np.outer(m_idx + 1.0, (2.0 * p_idx + 1.0) / (2.0 * self.N_p)))
        Ma = np.sin(np.pi * np.outer(m_idx + 1.0, (2.0 * p_idx + 1.0) / (2.0 * self.N_p)))
        self.Mb = tf.constant(Mb.astype(np.float32), dtype=dtype)
        self.Ma = tf.constant(Ma.astype(np.float32), dtype=dtype)
        scale = (4.0 / np.pi) * (1.0 / (2.0 * m_idx + 1.0))
        self.scale = tf.constant(scale.astype(np.float32), dtype=dtype)

        n = np.arange(1, self.N_m + 1, dtype=np.float32)
        factor = 1.0 / (n * np.pi)
        factor2 = factor ** 2
        self.factor_col = tf.constant(factor[:, None].astype(np.float32), dtype=dtype)
        self.factor2_col = tf.constant(factor2[:, None].astype(np.float32), dtype=dtype)

        C = np.cos(np.pi * np.outer(n, self.xi_np + 1.0))
        S = np.sin(np.pi * np.outer(n, self.xi_np + 1.0))
        self.C = tf.constant(C.astype(np.float32), dtype=dtype)
        self.S = tf.constant(S.astype(np.float32), dtype=dtype)
        self.Cm1 = self.C - 1.0
        self.one_m_C = 1.0 - self.C

    def coeffs(self, W):
        bn_num = tf.einsum("mn,bjn->bjm", self.Mb, W)
        an_num = tf.einsum("mn,bjn->bjm", self.Ma, W)
        b_n = bn_num * self.scale[None, None, :]
        a_n = an_num * self.scale[None, None, :]
        a0 = tf.reduce_mean(W, axis=-1)
        return a0, a_n, b_n

    def __call__(self, W_batch, x0_batch, xdot0_batch):
        """ADA-F forward.

        W_batch     : (B, state_dim, N_p) panel weights
        x0_batch    : (B, state_dim) IC anchor for affine free part
        xdot0_batch : (B, state_dim) derivative IC anchor = f(x0, theta)
        """
        W = tf.convert_to_tensor(W_batch, dtype=self.dtype)
        x0_batch = tf.convert_to_tensor(x0_batch, dtype=self.dtype)
        xdot0_batch = tf.convert_to_tensor(xdot0_batch, dtype=self.dtype)

        a0, a_n, b_n = self.coeffs(W)

        # ξ shifted by +1 so that both `xdot_free` and `x_free` symbolically
        # vanish at ξ=-1 (i.e., at t=0) → hard IC enforcement.
        xi_p1 = (self.xi + 1.0)[None, None, :]
        xddot_local = (
            0.5 * a0[:, :, None]
            + 0.5 * tf.einsum("bjn,nt->bjt", a_n, self.C)
            + 0.5 * tf.einsum("bjn,nt->bjt", b_n, self.S)
        )

        xdot_free_local = (
            0.5 * a0[:, :, None] * xi_p1
            + tf.einsum("bjn,nt->bjt", -0.5 * self.factor_col[None, None, :, 0] * b_n, self.Cm1)
            + tf.einsum("bjn,nt->bjt",  0.5 * self.factor_col[None, None, :, 0] * a_n, self.S)
        )

        x_free_local = (
            (a0[:, :, None] / 4.0) * tf.square(self.xi + 1.0)[None, None, :]
            + tf.einsum(
                "bjn,nt->bjt",
                -0.5 * self.factor_col[None, None, :, 0] * b_n,
                self.factor_col * self.S - (self.xi[None, :] + 1.0),
            )
            + tf.einsum(
                "bjn,nt->bjt",
                0.5 * self.factor2_col[None, None, :, 0] * a_n,
                self.one_m_C,
            )
        )

        xdddot_local = (
            0.5 * tf.einsum("bjn,nt->bjt", -a_n / self.factor_col[None, None, :, 0], self.S)
            + 0.5 * tf.einsum("bjn,nt->bjt",  b_n / self.factor_col[None, None, :, 0], self.C)
        )

        # Chain-rule consistent scaling (dx/dt = ẋ, dẋ/dt = ẍ):
        #   x  = x0 + ẋ0·t + x_free       (no s₁² factor — derived from
        #                                  d/dt[x_free(ξ)] = s₁·xdot_free)
        #   ẋ  = ẋ0 + s₁·xdot_free
        #   ẍ  = s₁²·xddot_local
        s1 = self.time_scale
        xdot = xdot0_batch[:, :, None] + s1 * xdot_free_local
        x = (x0_batch[:, :, None]
             + xdot0_batch[:, :, None] * self.t_local[None, None, :]
             + x_free_local)
        xddot = (s1 ** 2) * xddot_local
        xdddot = (s1 ** 3) * xdddot_local

        x = tf.transpose(x, [0, 2, 1])
        xdot = tf.transpose(xdot, [0, 2, 1])
        xddot = tf.transpose(xddot, [0, 2, 1])
        xdddot = tf.transpose(xdddot, [0, 2, 1])

        return {
            "x": x,
            "xdot": xdot,
            "xddot": xddot,
            "xdddot": xdddot,
            "x_end": x[:, -1, :],
            "t_local": self.t_local,
        }
