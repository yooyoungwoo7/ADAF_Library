#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lpa_basis.py
============================================================
W-panel Legendre Polynomial Approximation basis.

Network outputs panel weights W ∈ ℝ^{state_dim × N_p} on N_p equal panels of
x ∈ [-1, 1]. Legendre coefficients are obtained via integration:

    a₀ = (1/N_p) Σ_p W_p
    Aₙ = Σ_p coef_mat[n, p] · W_p    with
         coef_mat[n, p] = (2n+1)/2 · ∫_{x_p}^{x_{p+1}} Pₙ(x) dx

Forward (per state):
    xdot(x) = a₀ · P₀(x) + Σ_{n=1..M} Aₙ · Pₙ(x)
    x(x)    = x0 + (1/time_scale) · [a₀ · I1₀(x) + Σ_n Aₙ · I1ₙ(x)]

I1ₙ(-1) = 0 is enforced symbolically, so x(0) = x0 holds exactly.
"""

import numpy as np
import tensorflow as tf

from config import TF_DTYPE, DT_SEG, NT_SEG, GAMMA, MAX_ORDER, LPA_N_P
from utils.poly_utils import (
    build_legendre_symbolic_tables,
    pad_coeff_dict_to_common_width,
    get_legendre_panel_coefs_sympy_on_custom_panels,
    differentiate_poly_matrix,
)


class BatchLPABasis(tf.Module):
    def __init__(self, T_seg=DT_SEG, Nt=NT_SEG, gamma=GAMMA,
                 max_order=MAX_ORDER, N_p=LPA_N_P, dtype=None, name=None):
        super().__init__(name=name)
        if dtype is None:
            dtype = getattr(tf, TF_DTYPE)
        self.dtype = dtype
        self.T_seg = float(T_seg)
        self.Nt = int(Nt)
        self.gamma = float(gamma)
        self.max_order = int(max_order)
        self.N_p = int(N_p)
        self.coef_dim = self.N_p

        self.t_local_np = np.linspace(0.0, self.T_seg, self.Nt, dtype=np.float32)
        self.xgrid_np = (-1.0 + 2.0 * self.gamma * (self.t_local_np / self.T_seg)).astype(np.float32)
        self.x = tf.constant(self.xgrid_np, dtype=dtype)
        self.time_scale = tf.constant(2.0 * self.gamma / self.T_seg, dtype=dtype)

        P_dict, I1_dict, _, _ = build_legendre_symbolic_tables(max_order=self.max_order, dtype=np.float64)
        common_width = self.max_order + 4
        self.P_poly_mat = tf.constant(pad_coeff_dict_to_common_width(P_dict, self.max_order, common_width, dtype=np.float64), dtype=dtype)
        self.I1_poly_mat = tf.constant(pad_coeff_dict_to_common_width(I1_dict, self.max_order, common_width, dtype=np.float64), dtype=dtype)
        self.Pd_poly_mat = tf.constant(differentiate_poly_matrix(self.P_poly_mat.numpy()), dtype=dtype)

        panel_edges_np = np.linspace(-1.0, 1.0, self.N_p + 1).astype(np.float64)
        coef_rows = [
            get_legendre_panel_coefs_sympy_on_custom_panels(order=n, panel_edges=panel_edges_np, dtype=np.float64)
            for n in range(1, self.max_order + 1)
        ]
        self.coef_mat = tf.constant(np.stack(coef_rows, axis=0), dtype=dtype)  # (M, N_p)
        self.mean_vec = tf.constant(np.ones((self.N_p,), dtype=np.float64) / float(self.N_p), dtype=dtype)

        order_idx = tf.constant(np.arange(1, self.max_order + 1), dtype=tf.int32)
        self.P_cache = self._build_basis_cache(self.P_poly_mat, order_idx)
        self.I1_cache = self._build_basis_cache(self.I1_poly_mat, order_idx)
        self.Pd_cache = self._build_basis_cache(self.Pd_poly_mat, order_idx)
        self.P0_cache = self._eval_poly_matrix(self.P_poly_mat[0:1], self.x)[0]
        self.I1_0_cache = self._eval_poly_matrix(self.I1_poly_mat[0:1], self.x)[0]
        self.Pd0_cache = self._eval_poly_matrix(self.Pd_poly_mat[0:1], self.x)[0]

        B_panel = (self.P0_cache[:, None] * self.mean_vec[None, :]
                   + tf.einsum("ot,on->tn", self.P_cache, self.coef_mat))
        self.B_panel = B_panel  # (Nt, N_p)

    def _eval_poly_matrix(self, poly_mat, x):
        poly_mat = tf.convert_to_tensor(poly_mat, dtype=self.dtype)
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        y = tf.zeros((tf.shape(poly_mat)[0], tf.shape(x)[0]), dtype=self.dtype)
        coeffs_rev = tf.reverse(poly_mat, axis=[1])
        for c in tf.unstack(coeffs_rev, axis=1):
            y = y * x[None, :] + c[:, None]
        return y

    def _build_basis_cache(self, poly_mat, order_idx):
        sub_mat = tf.gather(poly_mat, order_idx, axis=0)
        return self._eval_poly_matrix(sub_mat, self.x)

    def __call__(self, W_batch, x0_batch, xdot0_batch=None):
        """W-panel forward.

        W_batch     : (B, state_dim, N_p) panel weights
        x0_batch    : (B, state_dim) hard IC
        xdot0_batch : unused (LPA enforces x(0)=x0 already)
        """
        W_batch = tf.convert_to_tensor(W_batch, dtype=self.dtype)
        x0_batch = tf.convert_to_tensor(x0_batch, dtype=self.dtype)

        a0 = tf.reduce_sum(W_batch * self.mean_vec[None, None, :], axis=-1)    # (B, J)
        A = tf.einsum("on,bjn->bjo", self.coef_mat, W_batch)                   # (B, J, M)

        xdot_val = (a0[:, :, None] * self.P0_cache[None, None, :]
                    + tf.einsum("bjo,ot->bjt", A, self.P_cache))
        x = ((a0[:, :, None] * self.I1_0_cache[None, None, :]
              + tf.einsum("bjo,ot->bjt", A, self.I1_cache))
             / self.time_scale
             + x0_batch[:, :, None])
        xddot_val = (a0[:, :, None] * self.Pd0_cache[None, None, :]
                     + tf.einsum("bjo,ot->bjt", A, self.Pd_cache))

        x = tf.transpose(x, [0, 2, 1])
        xdot = tf.transpose(xdot_val, [0, 2, 1])
        xddot = tf.transpose(self.time_scale * xddot_val, [0, 2, 1])
        xdddot = tf.zeros_like(xddot)

        return {
            "x": x,
            "xdot": xdot,
            "xddot": xddot,
            "xdddot": xdddot,
            "x_end": x[:, -1, :],
            "t_local": tf.constant(self.t_local_np, dtype=self.dtype),
        }
