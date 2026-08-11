#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/plotting.py
============================================================
Generic plotting helpers. Time unit and per-state labels are passed in by
the caller (see eval/sweep_plots, which read them from the active problem).
"""

from utils._style import apply_style, tight_x
apply_style()

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, save_path=None):
    has_lr = "lr" in history and len(history["lr"]) > 0
    loss_keys = [k for k in history if k != "lr" and len(history.get(k, [])) > 0]

    if has_lr:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))

    n_ep = max((len(history[k]) for k in loss_keys), default=0)
    x_ep = np.arange(n_ep)
    for k in loss_keys:
        ax1.plot(x_ep[:len(history[k])], history[k], label=k)
    ax1.set_yscale("log")
    ax1.set_ylabel("loss")
    ax1.set_title("Operator training history")
    ax1.legend(fontsize=10)
    if n_ep >= 2:
        tight_x(ax1, x_ep)

    if has_lr:
        lr_arr = history["lr"]
        ax2.plot(np.arange(len(lr_arr)), lr_arr, color="C1", label="learning rate")
        ax2.set_yscale("log")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("LR")
        ax2.legend(fontsize=10)
        if len(lr_arr) >= 2:
            tight_x(ax2, np.arange(len(lr_arr)))
    else:
        ax1.set_xlabel("epoch")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _ref_ylim(y_ref, pad_ratio=0.08, min_pad=1e-6):
    y_ref = np.asarray(y_ref, dtype=float)
    if y_ref.size == 0:
        return None
    y_min = float(np.nanmin(y_ref))
    y_max = float(np.nanmax(y_ref))
    span = y_max - y_min
    if not np.isfinite(span):
        return None
    if span < min_pad:
        center = 0.5 * (y_max + y_min)
        pad = max(abs(center) * pad_ratio, min_pad)
        return center - pad, center + pad
    pad = max(span * pad_ratio, min_pad)
    return y_min - pad, y_max + pad


def plot_state_vs_reference(t, pred, ref, labels, title_prefix="",
                            save_path=None, subtitle=None, time_unit="",
                            extras=None, theta=None, time_factor=1.0):
    """Per-state rows + optional auxiliary rows (extras).

    extras: list of dicts {label, unit, fn(x, theta), kind ∈ {"derived","constant"}}.
            Requires theta when 'constant' traces are present.
    time_factor: multiplier applied to t for display (e.g. 60.0 for h→min).
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    t = np.asarray(t) * float(time_factor)
    extras = extras or []
    n_state = len(labels)
    n_total = n_state + len(extras)
    fig, axes = plt.subplots(n_total, 1, figsize=(10, 2.5 * n_total + 0.5), sharex=True)
    if n_total == 1:
        axes = [axes]

    for j, lab in enumerate(labels):
        ax = axes[j]
        ax.plot(t, ref[:, j], color="k", lw=1.8, label="reference")
        ax.plot(t, pred[:, j], "--", color="C3", lw=1.5, label="operator")
        ylim = _ref_ylim(ref[:, j])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(lab)
        ax.legend(fontsize=10, loc="best")
        tight_x(ax, t)

    for k, spec in enumerate(extras):
        ax = axes[n_state + k]
        label = spec["label"]; unit = spec.get("unit", "")
        fn = spec["fn"]; kind = spec.get("kind", "derived")
        if kind == "derived":
            ax.plot(t, fn(ref,  theta), color="k",  lw=1.8, label="reference")
            ax.plot(t, fn(pred, theta), "--", color="C3", lw=1.5, label="operator")
        else:
            ax.plot(t, fn(pred, theta), color="C0", lw=1.5, label=label)
        ax.set_ylabel(label + (f" [{unit}]" if unit else ""))
        tight_x(ax, t)

    axes[0].set_title(title_prefix + " state trajectory", fontsize=16, pad=28)
    if subtitle is not None:
        axes[0].text(0.5, 1.06, subtitle, transform=axes[0].transAxes,
                     ha="center", va="bottom", fontsize=10)
    xlabel = r"$t$" + (f" [{time_unit}]" if time_unit else "")
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_profile(t, residual, labels, save_path=None, time_unit="", time_factor=1.0):
    residual = np.asarray(residual)
    t = np.asarray(t) * float(time_factor)
    palette = ["C0", "C1", "C3", "C4"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, lab in enumerate(labels):
        ax.plot(t, residual[:, j], color=palette[j % len(palette)], lw=1.4, label=lab)
    ax.axhline(0.0, color="k", lw=0.3)
    xlabel = r"$t$" + (f" [{time_unit}]" if time_unit else "")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("residual")
    ax.set_title("Residual profile")
    ax.legend(fontsize=10, loc="best")
    tight_x(ax, t)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
