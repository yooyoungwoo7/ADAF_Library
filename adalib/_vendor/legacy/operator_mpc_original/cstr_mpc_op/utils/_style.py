#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/_style.py
============================================================
Shared matplotlib style for all triple-tank evaluation plots.
"""

from __future__ import annotations

_APPLIED = False


def apply_style():
    global _APPLIED
    if _APPLIED:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          14,
        "figure.dpi":         400,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top":          True,
        "ytick.right":        True,
        "axes.grid":          False,
        "axes.unicode_minus": False,
    })
    _APPLIED = True


def axes_zero_lines(ax, x=True, y=True, color="k", lw=0.3):
    if x:
        ax.axhline(0.0, color=color, lw=lw)
    if y:
        ax.axvline(0.0, color=color, lw=lw)


def tight_x(ax, x):
    """Clamp x-axis to [x[0], x[-1]] — no left/right padding."""
    import numpy as _np
    x = _np.asarray(x)
    if x.size >= 2:
        ax.set_xlim(float(x[0]), float(x[-1]))
