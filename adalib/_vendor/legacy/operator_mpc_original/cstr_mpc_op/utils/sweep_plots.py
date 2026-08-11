#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/sweep_plots.py
============================================================
Aggregate validation plots — fully driven by problem metadata
(state_labels, time_unit, nominal_input(), sweep_specs(), random_input(),
case_subtitle()). Same code path for all problems.
"""

from __future__ import annotations

import numpy as np

from utils._style import apply_style, tight_x
apply_style()

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from config import N_SEG, STATE_DIM, PARAM_DIM


def _ylabel(state_label, unit):
    return state_label + (f" [{unit}]" if unit else "")


def _rollout_and_ref(learner, x_input):
    r = learner.rollout_full_trajectory(x_input, n_seg=N_SEG)
    ref = learner.problem.solve_reference(
        theta=x_input[STATE_DIM:STATE_DIM + PARAM_DIM],
        x0=x_input[:STATE_DIM],
        t_grid=r["t"],
    )
    return r["t"], r["x"], ref


def plot_param_sweeps(learner, save_path, n=8, nominal=None):
    problem = learner.problem
    specs = problem.sweep_specs()
    if not specs:
        return plot_random_cases(learner, save_path, n_cases=4, seed=0)

    base = problem.nominal_input() if nominal is None else np.asarray(nominal, dtype=np.float32)
    n_states = problem.state_dim
    extras = problem.extra_plot_traces() or []
    n_extra = len(extras)
    n_rows = n_states + n_extra
    n_cols = len(specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.6 * n_rows + 0.5))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]
    cmap = plt.get_cmap("plasma")
    units = problem.state_units()
    time_unit = problem.time_unit
    xlabel = r"$t$" + (f" [{time_unit}]" if time_unit else "")
    tfac = float(getattr(problem, "time_factor", 1.0))

    for col, (label, vals, idx) in enumerate(specs):
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
        trajs_pred, trajs_ref, thetas, t_common = [], [], [], None
        for v in vals:
            x_in = base.copy()
            x_in[idx] = float(v)
            t_raw, pred, ref = _rollout_and_ref(learner, x_in)
            trajs_pred.append(pred)
            trajs_ref.append(ref)
            thetas.append(x_in[STATE_DIM:STATE_DIM + PARAM_DIM])
            t_common = t_raw * tfac

        for row in range(n_states):
            ax = axes[row, col]
            for v, pred, ref in zip(vals, trajs_pred, trajs_ref):
                c = cmap(norm(float(v)))
                ax.plot(t_common, ref[:, row], color=c, lw=1.5, alpha=0.8)
                ax.plot(t_common, pred[:, row], color=c, lw=1.5, linestyle="--")
            tight_x(ax, t_common)
            if row == 0:
                ax.set_title(f"sweep: {label}")
            if col == 0:
                ax.set_ylabel(_ylabel(problem.state_plot_labels()[row], units[row]))

        # extra rows
        for j, spec in enumerate(extras):
            row = n_states + j
            ax = axes[row, col]
            ex_label = spec["label"]; ex_unit = spec.get("unit", "")
            fn = spec["fn"]; kind = spec.get("kind", "derived")
            for v, pred, ref, th in zip(vals, trajs_pred, trajs_ref, thetas):
                c = cmap(norm(float(v)))
                if kind == "derived":
                    ax.plot(t_common, fn(ref,  th), color=c, lw=1.5, alpha=0.8)
                    ax.plot(t_common, fn(pred, th), color=c, lw=1.5, linestyle="--")
                else:
                    ax.plot(t_common, fn(pred, th), color=c, lw=1.5)
            tight_x(ax, t_common)
            if col == 0:
                ax.set_ylabel(_ylabel(ex_label, ex_unit))

        axes[n_rows - 1, col].set_xlabel(xlabel)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:, col].tolist(), label=label, shrink=0.92, pad=0.02)

    fig.suptitle(f"{problem.name} — parameter sweeps (operator dashed vs reference solid)",
                 fontsize=15, y=1.00)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_case_grid(x_inputs, learner, save_path, suptitle):
    problem = learner.problem
    n_states = problem.state_dim
    extras = problem.extra_plot_traces() or []
    n_extra = len(extras)
    n_rows = n_states + n_extra
    n_cases = len(x_inputs)
    fig, axes = plt.subplots(n_rows, n_cases,
                             figsize=(4.2 * n_cases, 2.5 * n_rows + 1),
                             sharex=True)
    if n_rows == 1:
        axes = axes[None, :]
    if n_cases == 1:
        axes = axes[:, None]
    units = problem.state_units()
    time_unit = problem.time_unit
    xlabel = r"$t$" + (f" [{time_unit}]" if time_unit else "")
    tfac = float(getattr(problem, "time_factor", 1.0))

    for col, x_in in enumerate(x_inputs):
        t_raw, pred, ref = _rollout_and_ref(learner, x_in)
        t = t_raw * tfac
        theta = x_in[STATE_DIM:STATE_DIM + PARAM_DIM]
        subtitle = problem.case_subtitle(x_in)

        # state rows
        for row in range(n_states):
            ax = axes[row, col]
            ax.plot(t, ref[:, row], color="k", lw=1.8,
                    label="reference" if (row == 0 and col == 0) else None)
            ax.plot(t, pred[:, row], color="C3", lw=1.5, linestyle="--",
                    label="operator" if (row == 0 and col == 0) else None)
            tight_x(ax, t)
            if col == 0:
                ax.set_ylabel(_ylabel(problem.state_plot_labels()[row], units[row]))
            if row == 0:
                ax.set_title(subtitle, fontsize=9)

        # extra rows (auxiliary traces)
        for j, spec in enumerate(extras):
            row = n_states + j
            ax = axes[row, col]
            label = spec["label"]
            unit  = spec.get("unit", "")
            fn    = spec["fn"]
            kind  = spec.get("kind", "derived")
            if kind == "derived":
                ax.plot(t, fn(ref,  theta), color="k",  lw=1.8)
                ax.plot(t, fn(pred, theta), color="C3", lw=1.5, linestyle="--")
            else:  # "constant" — single horizontal trace from theta only
                ax.plot(t, fn(pred, theta), color="C0", lw=1.5)
            tight_x(ax, t)
            if col == 0:
                ax.set_ylabel(_ylabel(label, unit))

        # bottom row x-label
        axes[n_rows - 1, col].set_xlabel(xlabel)
        if col == 0:
            axes[0, col].legend(fontsize=9, loc="best")

    fig.suptitle(suptitle, fontsize=13, y=1.00)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_val_comparison(learner, fullcase_path, save_path, n_cases=4, seed=0):
    full = np.load(fullcase_path)
    X0 = np.asarray(full["X0"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(X0.shape[0], size=min(n_cases, X0.shape[0]), replace=False)
    x_inputs = [X0[i] for i in idxs]
    _plot_case_grid(
        x_inputs, learner, save_path,
        suptitle=f"{learner.problem.name} — operator vs reference ({N_SEG} segments)",
    )


def plot_random_cases(learner, save_path, n_cases=4, seed=None):
    rng = np.random.default_rng(seed)
    x_inputs = learner.problem.diverse_random_inputs(n_cases, rng)
    _plot_case_grid(
        x_inputs, learner, save_path,
        suptitle=f"{learner.problem.name} — random inputs ({n_cases} cases, {N_SEG} segments)",
    )
