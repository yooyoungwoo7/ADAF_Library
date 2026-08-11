#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp


# ============================================================
# Symbolic helpers
# ============================================================
def sympy_poly_to_numpy_coeffs(expr, x_sym, dtype=np.float32):
    poly = sp.Poly(sp.expand(expr), x_sym)
    coeff_dict = poly.as_dict()
    max_deg = poly.degree()
    coeffs = np.zeros((max_deg + 1,), dtype=np.float64)
    for k, v in coeff_dict.items():
        coeffs[k[0]] = float(v)
    return coeffs.astype(dtype)


def build_legendre_symbolic_tables(max_order=6, dtype=np.float32):
    x = sp.symbols("x", real=True)

    P_coeffs = {}
    I1_coeffs = {}
    I2_coeffs = {}
    I3_coeffs = {}

    for n in range(max_order + 1):
        Pn = sp.legendre(n, x)

        I1 = sp.integrate(Pn, x)
        I1 = sp.expand(I1 - I1.subs(x, -1))

        I2 = sp.integrate(I1, x)
        I2 = sp.expand(I2 - I2.subs(x, -1))

        I3 = sp.integrate(I2, x)
        I3 = sp.expand(I3 - I3.subs(x, -1))

        P_coeffs[n] = sympy_poly_to_numpy_coeffs(Pn, x, dtype=dtype)
        I1_coeffs[n] = sympy_poly_to_numpy_coeffs(I1, x, dtype=dtype)
        I2_coeffs[n] = sympy_poly_to_numpy_coeffs(I2, x, dtype=dtype)
        I3_coeffs[n] = sympy_poly_to_numpy_coeffs(I3, x, dtype=dtype)

    return P_coeffs, I1_coeffs, I2_coeffs, I3_coeffs


def pad_coeff_dict_to_common_width(coeff_dict, max_order, common_width, dtype=np.float32):
    mat = np.zeros((max_order + 1, common_width), dtype=dtype)
    for n in range(max_order + 1):
        c = coeff_dict[n]
        mat[n, :len(c)] = c
    return mat


def get_legendre_panel_coefs_sympy_on_custom_panels(order, panel_edges, dtype=np.float32):
    x = sp.symbols("x", real=True)
    P = sp.legendre(order, x)
    Pint = sp.integrate(P, x)

    vals = np.array([float(Pint.subs(x, s)) for s in panel_edges], dtype=np.float64)
    coefs = vals[1:] - vals[:-1]
    coefs *= (2.0 * order + 1.0) / 2.0
    return coefs.astype(dtype)


def differentiate_poly_matrix(poly_mat_np):
    poly_mat_np = np.asarray(poly_mat_np, dtype=np.float64)
    n_basis, width = poly_mat_np.shape
    out = np.zeros_like(poly_mat_np)
    for k in range(1, width):
        out[:, k - 1] = k * poly_mat_np[:, k]
    return out
