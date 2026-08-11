"""
adalib/utils/plotting.py
Publication-quality plotting utilities for ADALib forward, operator, and MPC results.

Usage
-----
    import adalib

    adalib.utils.set_adalib_plot_style()

    fig, axes = adalib.utils.plot_forward_result(result, reference=..., save_path=...)
    fig, axes, metrics = adalib.utils.plot_operator_result(result, ...)
    fig, axes = adalib.utils.plot_mpc_result(result, ...)

Headless note
-------------
    Call ``matplotlib.use("Agg")`` *before* importing matplotlib/pyplot if running
    in a headless environment (CI, server, etc.).
"""
from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ── Palette ──────────────────────────────────────────────────────────────────
_C_REF   = "k"          # black — reference / ground truth
_C_PRED  = "#C0392B"    # dark red — ADA/operator prediction (single-state mode)
_C_CTRL  = "#534AB7"    # indigo — control input
_C_COST  = "#27AE60"    # green — cost / objective
_C_TGT   = "red"        # red dashed — setpoint target
_C_ERR   = "#E67E22"    # orange — error

# Per-state colors for grouped plots (skips C2=green, reserved for cost/error)
_STATE_COLORS = ["C0", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


# ── Global style ──────────────────────────────────────────────────────────────

def set_adalib_plot_style(style: str = "sans") -> None:
    """Apply ADALib matplotlib rcParams.

    Parameters
    ----------
    style
        ``"sans"`` (default) — clean sans-serif style with grid.
        ``"serif"`` — Times New Roman, inward ticks, minor ticks, no grid
        (matches the academic paper style from operator_lib).
    """
    common = {
        "legend.framealpha":  0.7,
        "lines.linewidth":    1.8,
        "savefig.bbox":       "tight",
        "axes.unicode_minus": False,
    }
    if style == "serif":
        plt.rcParams.update({
            **common,
            "font.family":           "serif",
            "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset":      "custom",
            "mathtext.rm":           "Times New Roman",
            "mathtext.it":           "Times New Roman:italic",
            "mathtext.bf":           "Times New Roman:bold",
            "font.size":             16,
            "axes.titlesize":        14,
            "axes.labelsize":        16,
            "xtick.labelsize":       14,
            "ytick.labelsize":       14,
            "legend.fontsize":       13,
            "figure.titlesize":      18,
            "figure.dpi":            150,
            "savefig.dpi":           400,
            "xtick.direction":       "out",
            "ytick.direction":       "out",
            "xtick.minor.visible":   False,
            "ytick.minor.visible":   False,
            "xtick.top":             False,
            "ytick.right":           False,
            "axes.grid":             False,
            "axes.spines.top":       True,
            "axes.spines.right":     True,
        })
    else:
        plt.rcParams.update({
            **common,
            "font.family":           "sans-serif",
            "font.size":             11,
            "axes.titlesize":        11,
            "axes.labelsize":        11,
            "xtick.labelsize":       10,
            "ytick.labelsize":       10,
            "legend.fontsize":       9,
            "figure.titlesize":      13,
            "figure.dpi":            100,
            "savefig.dpi":           150,
            "grid.alpha":            0.3,
            "axes.grid":             True,
            "axes.spines.top":       False,
            "axes.spines.right":     False,
        })


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """Plot operator training loss (and optional learning-rate) curves.

    Parameters
    ----------
    history
        Dict with keys like ``"train_loss"``, ``"val_loss"``, and optionally
        ``"lr"``.  Values are lists of floats, one per epoch.
    save_path
        If given, save figure here.
    show
        If True, call ``plt.show()``.

    Returns
    -------
    (fig, axes)
    """
    has_lr = "lr" in history and len(history.get("lr", [])) > 0
    loss_keys = [k for k in history if k != "lr" and len(history.get(k, [])) > 0]

    if has_lr:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        axes = np.array([ax1, ax2])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
        axes = np.array([ax1])

    n_ep = max((len(history[k]) for k in loss_keys), default=0)
    x_ep = np.arange(n_ep)
    for k in loss_keys:
        ax1.semilogy(x_ep[:len(history[k])], history[k], label=k)
    ax1.set_ylabel("loss")
    ax1.set_title("Operator training history")
    ax1.legend(fontsize=10)
    if n_ep >= 2:
        _tight_x(ax1, x_ep)

    if has_lr:
        lr_arr = np.asarray(history["lr"], dtype=float)
        ax2.semilogy(np.arange(len(lr_arr)), lr_arr, color="C1", label="learning rate")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("LR")
        ax2.legend(fontsize=10)
        if len(lr_arr) >= 2:
            _tight_x(ax2, np.arange(len(lr_arr)))
    else:
        ax1.set_xlabel("epoch")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tight_x(ax: Any, t: np.ndarray) -> None:
    """Set x-axis limits to [t[0], t[-1]]."""
    ax.set_xlim(float(t[0]), float(t[-1]))


def _open_ax(ax: Any) -> None:
    """Enforce open-box style (left+bottom only, outward major ticks, no minor ticks).

    Called per-axes so the style is applied regardless of rcParams state.
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="out")
    ax.minorticks_off()


def _safe_ylim(data: np.ndarray, pad: float = 0.08, min_pad: float = 1e-6
               ) -> Optional[Tuple[float, float]]:
    """Compute symmetrically padded y-limits that avoid flat axes."""
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return None
    y_min, y_max = float(finite.min()), float(finite.max())
    span = y_max - y_min
    if span < min_pad:
        center = 0.5 * (y_min + y_max)
        p = max(abs(center) * pad, min_pad)
        return center - p, center + p
    return y_min - span * pad, y_max + span * pad


def _ensure_time_state(y: np.ndarray, t_len: int) -> np.ndarray:
    """Normalize y to shape (T, n_state).

    Accepts (T, n_state), (n_state, T), or 1-D arrays.
    Ambiguous square arrays are left unchanged.
    """
    if y.ndim == 1:
        return y[:, np.newaxis]
    if y.ndim == 2:
        if y.shape[0] == t_len and y.shape[1] != t_len:
            return y          # already (T, n_state)
        if y.shape[1] == t_len and y.shape[0] != t_len:
            return y.T        # (n_state, T) → (T, n_state)
        # Square or ambiguous: try to guess from t_len
        if y.shape[0] == t_len:
            return y
        return y.T
    raise ValueError(f"Unexpected y.ndim={y.ndim}")


def _parse_reference_item(
    ref: Any,
    t_op: np.ndarray,
    n_state: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalise one reference item to (t_ref (T,), y_ref (T, n_state))."""
    if ref is None:
        return None, None

    # scipy OdeResult
    if hasattr(ref, "t") and hasattr(ref, "y"):
        t_r = np.asarray(ref.t, dtype=float)
        y_r = _ensure_time_state(np.asarray(ref.y, dtype=float), len(t_r))
        return t_r, y_r

    # (t_array, y_array) tuple
    if isinstance(ref, (tuple, list)) and len(ref) == 2:
        try:
            t_r = np.asarray(ref[0], dtype=float)
            y_r = _ensure_time_state(np.asarray(ref[1], dtype=float), len(t_r))
            return t_r, y_r
        except (TypeError, ValueError):
            pass

    # callable: reference(t) → array of shape (T,) or (T, n_state)
    if callable(ref):
        t_r = np.asarray(t_op, dtype=float)
        y_raw = np.asarray(ref(t_r), dtype=float)
        y_r   = _ensure_time_state(y_raw, len(t_r))
        if y_r.shape[1] < n_state:
            y_r = np.tile(y_r, (1, n_state))
        return t_r, y_r

    warnings.warn(
        f"Unrecognised reference type {type(ref).__name__!r}. "
        "Accepted: scipy OdeResult, (t, y) tuple, or callable(t)->y.",
        stacklevel=3,
    )
    return None, None


# ── 1. plot_forward_result ────────────────────────────────────────────────────

def plot_forward_result(
    result: Any,
    reference: Any = None,
    state_names: Optional[List[str]] = None,
    state_groups: Optional[List[List[int]]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Plot a forward ODE solve result.

    Parameters
    ----------
    result
        Object with ``result.solution.t`` (shape ``(Nt,)``) and
        ``result.solution.y`` (shape ``(n_state, Nt)``).
    reference
        * ``callable(t) -> array`` — evaluated on ``result.solution.t``
        * ``ndarray`` — shape ``(Nt,)`` or ``(n_state, Nt)``
        * ``None`` — no reference
    state_names
        Labels for each state (default: ``["y_0", "y_1", …]``).
    state_groups
        List of index groups to overlay on a single axes.  E.g.
        ``[[0, 1, 2]]`` puts all three states on one subplot.
        When ``None`` (default), one subplot per state.
    save_path
        If given, save figure here.
    show
        If True, call ``plt.show()``.
    title
        Figure suptitle (auto-generated if None, suppressed if ``""``).

    Returns
    -------
    (fig, axes)
    """
    t     = np.asarray(result.solution.t, dtype=float)
    y_raw = np.asarray(result.solution.y, dtype=float)
    y     = _ensure_time_state(y_raw, len(t))
    n_state = y.shape[1]

    names = list(state_names) if state_names else [f"$y_{{{i}}}$" for i in range(n_state)]

    t_ref, y_ref = _parse_reference_item(reference, t, n_state)
    has_ref = y_ref is not None
    if has_ref and y_ref.shape[1] < n_state:
        y_ref = np.tile(y_ref[:, :1], (1, n_state))

    _groups = [list(g) for g in state_groups] if state_groups is not None else None

    if _groups is not None:
        # ── Grouped mode: one row per group, states overlaid ──────────
        n_rows = len(_groups)
        fig, axes = plt.subplots(n_rows, 1, figsize=(5.5, 5.0 * n_rows), squeeze=False)

        for grp_idx, grp_indices in enumerate(_groups):
            ax = axes[grp_idx, 0]
            _open_ax(ax)

            for i_in_grp, s_idx in enumerate(grp_indices):
                lbl   = names[s_idx] if s_idx < len(names) else f"$y_{{{s_idx}}}$"
                short = lbl.split("[")[0].strip()
                color = _STATE_COLORS[s_idx % len(_STATE_COLORS)]
                ax.plot(t, y[:, s_idx], color=color, lw=1.8, zorder=3, label=short)
                if has_ref:
                    ref_lbl = "reference" if i_in_grp == 0 else "_nolegend_"
                    ax.plot(t_ref, y_ref[:, s_idx], color=_C_REF, lw=1.2,
                            ls="--", zorder=4, label=ref_lbl)

            ax.set_ylabel(_group_ylabel_with_unit(grp_indices, names))
            _tight_x(ax, t)

            # Asymmetric y-limits: extra top space for panel label ("a)", "b)", …)
            all_vals = np.concatenate(
                [y[:, s] for s in grp_indices]
                + ([y_ref[:, s] for s in grp_indices] if has_ref else [])
            )
            finite = all_vals[np.isfinite(all_vals)]
            if len(finite):
                vmin, vmax = float(finite.min()), float(finite.max())
                span = max(vmax - vmin, 1e-6)
                ax.set_ylim(vmin - span * 0.08, vmax + span * 0.10)

            if grp_idx == n_rows - 1:
                ax.set_xlabel("$t$")

    else:
        # ── Original mode: one row per state ──────────────────────────
        n_cols = 2 if has_ref else 1
        fig, axes = plt.subplots(
            n_state, n_cols,
            figsize=(6 * n_cols, 3 * n_state + 0.5),
            squeeze=False,
        )

        for i in range(n_state):
            lbl = names[i] if i < len(names) else f"$y_{{{i}}}$"
            ax  = axes[i, 0]
            _open_ax(ax)

            if has_ref:
                ax.plot(t_ref, y_ref[:, i], color=_C_REF, lw=1.8, zorder=3, label="reference")
            ax.plot(t, y[:, i], "--", color=_C_PRED, lw=1.5, zorder=4, label="ADA-F")
            ax.set_ylabel(lbl)
            _tight_x(ax, t)
            ylim = _safe_ylim(y[:, i])
            if ylim:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.legend(loc="best")
            if i == n_state - 1:
                ax.set_xlabel("$t$")

            if has_ref:
                err  = np.abs(y[:, i] - np.interp(t, t_ref, y_ref[:, i]))
                ax_e = axes[i, 1]
                _open_ax(ax_e)
                ax_e.semilogy(t, np.maximum(err, 1e-16), color=_C_ERR, lw=1.5)
                ax_e.set_ylabel(f"|err|  {lbl}")
                _tight_x(ax_e, t)
                if i == n_state - 1:
                    ax_e.set_xlabel("$t$")

    if title:
        fig.suptitle(title)
    elif title is None:
        fig.suptitle("Forward solve")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


# ── 2. plot_operator_result ───────────────────────────────────────────────────

def plot_operator_result(
    result: Any,
    system: Any = None,
    x0: Any = None,
    params: Any = None,
    control: Any = None,
    reference: Any = None,
    state_names: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    state_groups: Optional[List[List[int]]] = None,
    group_ylabels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    t_scale: float = 60.0,
    t_unit: str = "min",
) -> Tuple[Any, Any, Dict]:
    """Plot operator rollout result(s), optionally vs a reference trajectory.

    Parameters
    ----------
    result
        ``OperatorResult`` or list of ``OperatorResult``.
        Each result has ``.t`` (N_SEG+1,) and ``.y`` (N_SEG+1, n_state).
    system
        ADALib ODE system, used when ``reference="solve_ivp"``.
    x0
        Initial state(s) for reference simulation.  Single array or list of
        arrays matching ``result``.
    params
        ODE parameter(s) for reference simulation.  Single array or list.
        If ``len(params) > len(system.parameter_names)``, excess values are
        treated as the control ``u``.
    control
        Explicit control ``u`` for ``system.simulate()``.
        Overrides any auto-split from ``params``.
    reference
        * ``"solve_ivp"`` — simulate using ``system.simulate()`` (BDF)
        * scipy ``OdeResult`` or list thereof — pre-computed reference(s)
        * ``(t_ref, y_ref)`` tuple or list of tuples
        * ``callable(t) -> array`` or list of callables
        * ``None`` — no reference
    state_names
        Override state variable labels.
    labels
        Column titles for multi-case plots (default: "Case 1", "Case 2", …).
    state_groups
        Group multiple states onto one row.  E.g. ``[[0, 1], [2, 3]]`` puts
        C_A/C_B on row 0 and T_R/T_K on row 1.  Each state gets its own color
        from ``_STATE_COLORS``; the reference is a solid line and the operator
        prediction is a dashed line in the same color.
        ``None`` (default) uses one row per state.
    group_ylabels
        Custom y-axis labels for each group row.  Auto-generated from
        ``state_names`` when ``None``.
    save_path
        If given, save figure here.
    show
        If True, call ``plt.show()``.
    title
        Figure suptitle.

    Returns
    -------
    (fig, axes, metrics)
        ``metrics`` is a dict with key ``"l2_rel"`` (shape n_cases × n_state)
        and ``"max_abs"`` (shape n_cases × n_state) when a reference is given.
    """
    # Normalise to list
    if not isinstance(result, (list, tuple)):
        results = [result]
        x0_list     = [x0]
        params_list = [params]
        ctrl_list   = [control]
        ref_list    = [reference]
    else:
        results     = list(result)
        n = len(results)
        x0_list     = list(x0)    if isinstance(x0, (list, tuple)) else [x0]    * n
        params_list = list(params) if isinstance(params, (list, tuple)) else [params] * n
        ctrl_list   = list(control) if isinstance(control, (list, tuple)) else [control] * n
        ref_list    = list(reference) if isinstance(reference, (list, tuple)) else [reference] * n

    n_cases = len(results)

    # Try to get state names from first result / system
    if state_names is None:
        if system is not None and hasattr(system, "state_names"):
            state_names = list(system.state_names)
        elif hasattr(results[0], "metadata"):
            pass  # leave as None; will default below

    n_state = results[0].y.shape[1]
    names   = list(state_names) if state_names else [f"state {i}" for i in range(n_state)]
    col_labels = list(labels) if labels else [f"Case {i+1}" for i in range(n_cases)]

    # Resolve row layout
    if state_groups is not None:
        _groups = [list(g) for g in state_groups]
    else:
        _groups = None   # per-state mode
    n_rows_plot = len(_groups) if _groups is not None else n_state

    # Figure layout: n_rows_plot rows × n_cases cols
    row_h = 3.2 if _groups is not None else 3.0
    fig, axes = plt.subplots(
        n_rows_plot, n_cases,
        figsize=(5 * n_cases, row_h * n_rows_plot + 0.5),
        squeeze=False,
    )

    metrics: Dict = {"l2_rel": [], "max_abs": []}

    for col, (res, x0_c, p_c, ctrl_c, ref_c) in enumerate(
            zip(results, x0_list, params_list, ctrl_list, ref_list)):

        t_op  = np.asarray(res.t,  dtype=float)   # (N_SEG+1,)
        y_op  = np.asarray(res.y,  dtype=float)   # (N_SEG+1, n_state)
        t_min = t_op * t_scale

        # --- Parse reference ------------------------------------------------
        t_ref_d, y_ref_d = None, None

        if isinstance(ref_c, str) and ref_c == "solve_ivp":
            if system is None:
                warnings.warn(
                    'reference="solve_ivp" requires system argument.', stacklevel=2
                )
            elif x0_c is None:
                warnings.warn(
                    'reference="solve_ivp" requires x0 argument.', stacklevel=2
                )
            else:
                p_sim, u_sim = _split_params(system, p_c, ctrl_c)
                t_fine = np.linspace(t_op[0], t_op[-1], 500)
                try:
                    sol = system.simulate(
                        np.asarray(x0_c, dtype=float),
                        (float(t_op[0]), float(t_op[-1])),
                        u=u_sim, p=p_sim,
                        t_eval=t_fine,
                        method="BDF",
                    )
                    t_ref_d = sol.t
                    y_ref_d = _ensure_time_state(np.asarray(sol.y, dtype=float), len(sol.t))
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"solve_ivp reference failed: {exc}", stacklevel=2)
        else:
            t_ref_d, y_ref_d = _parse_reference_item(ref_c, t_op, n_state)
            if t_ref_d is not None:
                t_ref_d = t_ref_d * t_scale

        has_ref = y_ref_d is not None

        # --- Per-case metrics (always per-state regardless of grouping) -----
        case_l2 = [float("nan")] * n_state
        case_ma = [float("nan")] * n_state
        if has_ref:
            for s in range(n_state):
                y_r_at_op = np.interp(t_min, t_ref_d, y_ref_d[:, s])
                diff = y_op[:, s] - y_r_at_op
                norm_r = np.linalg.norm(y_r_at_op) + 1e-12
                case_l2[s] = float(np.linalg.norm(diff) / norm_r)
                case_ma[s] = float(np.max(np.abs(diff)))
        metrics["l2_rel"].append(case_l2)
        metrics["max_abs"].append(case_ma)

        xlim_data = t_ref_d if has_ref else t_min

        # --- Plot (grouped or per-state) ------------------------------------
        if _groups is not None:
            _plot_grouped(
                col, axes, _groups, group_ylabels, names,
                t_min, y_op, t_ref_d, y_ref_d, has_ref,
                col_labels, n_rows_plot, n_cases, xlim_data, t_unit,
            )
        else:
            for row in range(n_state):
                ax  = axes[row, col]
                lbl = names[row] if row < len(names) else f"state {row}"

                ax.plot(t_min, y_op[:, row],
                        color=_C_PRED, lw=1.5, zorder=3,
                        label=f"operator (L2={case_l2[row]:.2e})" if has_ref else "operator")
                if has_ref:
                    ax.plot(t_ref_d, y_ref_d[:, row],
                            "k--", lw=1.0, zorder=4, label="reference")

                if col == 0:
                    ax.set_ylabel(lbl)
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=9, pad=4)
                    ax.legend(fontsize=8, loc="best")
                if row == n_state - 1:
                    ax.set_xlabel(f"$t$ [{t_unit}]")

                _tight_x(ax, xlim_data)
                ref_for_ylim = (y_ref_d[:, row] if has_ref else y_op[:, row])
                ylim = _safe_ylim(ref_for_ylim)
                if ylim:
                    ax.set_ylim(*ylim)

    suptitle = title or "Operator rollout"
    if results[0].metadata:
        sys_n = results[0].metadata.get("system_name", "")
        n_seg = results[0].metadata.get("n_seg", "")
        suptitle = title or f"{sys_n} — operator vs reference  ({n_seg} segments)"
    fig.suptitle(suptitle)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, metrics


def _plot_grouped(
    col, axes, groups, group_ylabels, names,
    t_min, y_op, t_ref_d, y_ref_d, has_ref,
    col_labels, n_rows_plot, n_cases, xlim_data, t_unit="min",
):
    """Render one case column with states grouped onto shared rows."""
    from matplotlib.lines import Line2D

    for grp_idx, grp_indices in enumerate(groups):
        ax = axes[grp_idx, col]

        for s_idx in grp_indices:
            color = _STATE_COLORS[s_idx % len(_STATE_COLORS)]
            ax.plot(t_min, y_op[:, s_idx], color=color, lw=1.5, zorder=3)
            if has_ref:
                ax.plot(t_ref_d, y_ref_d[:, s_idx], "k--", lw=1.0, zorder=4)

        # y-axis label (left column only)
        if col == 0:
            if group_ylabels and grp_idx < len(group_ylabels):
                ylabel = group_ylabels[grp_idx]
            else:
                ylabel = _group_ylabel(grp_indices, names)
            ax.set_ylabel(ylabel)

        # column title (top row only)
        if grp_idx == 0:
            ax.set_title(col_labels[col], fontsize=9, pad=4)

        # x-label (bottom row only)
        if grp_idx == n_rows_plot - 1:
            ax.set_xlabel(f"$t$ [{t_unit}]")

        _tight_x(ax, xlim_data)

        # y-limits: based on all reference data in the group
        ref_data = (
            np.concatenate([y_ref_d[:, s] for s in grp_indices])
            if has_ref else
            np.concatenate([y_op[:, s] for s in grp_indices])
        )
        ylim = _safe_ylim(ref_data)
        if ylim:
            ax.set_ylim(*ylim)

        # Legend (first case column only, per group row)
        if col == 0:
            state_handles = [
                Line2D([], [], color=_STATE_COLORS[i % len(_STATE_COLORS)], lw=1.5,
                       label=names[i].split("[")[0].strip() if i < len(names) else f"state {i}")
                for i in grp_indices
            ]
            if has_ref:
                ref_handle = Line2D([], [], color="k", ls="--", lw=1.0, label="reference")
                ax.legend(handles=state_handles + [ref_handle], fontsize=9, loc="best")
            else:
                ax.legend(handles=state_handles, fontsize=9, loc="best")


def _group_ylabel(indices: List[int], names: List[str]) -> str:
    """Auto-generate ylabel for a group: short state names without units, joined by ', '."""
    parts = []
    for i in indices:
        n = names[i] if i < len(names) else f"state {i}"
        parts.append(n.split("[")[0].strip())
    return ", ".join(parts)


def _group_ylabel_with_unit(indices: List[int], names: List[str]) -> str:
    """Like _group_ylabel but appends the unit from the first state that has one."""
    parts = []
    unit = ""
    for i in indices:
        n = names[i] if i < len(names) else f"state {i}"
        parts.append(n.split("[")[0].strip())
        if not unit and "[" in n:
            unit = n[n.index("["):]
    ylabel = ", ".join(parts)
    if unit:
        ylabel += " " + unit
    return ylabel


def _split_params(
    system: Any,
    params: Any,
    control: Any,
) -> Tuple[Optional[np.ndarray], Any]:
    """Split params into (p, u) based on system.parameter_names length."""
    if control is not None:
        return (np.asarray(params, dtype=float) if params is not None else None), control

    if params is None:
        return None, None

    p_arr = np.asarray(params, dtype=float)
    n_sys_p = len(getattr(system, "parameter_names", []))
    if n_sys_p > 0 and len(p_arr) > n_sys_p:
        p = p_arr[:n_sys_p]
        u_arr = p_arr[n_sys_p:]
        u = float(u_arr[0]) if len(u_arr) == 1 else u_arr
        return p, u
    return p_arr, None


# ── 3. plot_mpc_result ────────────────────────────────────────────────────────

def plot_mpc_result(
    result: Any,
    state_names: Optional[List[str]] = None,
    control_names: Optional[List[str]] = None,
    target: Any = None,
    labels: Optional[List[str]] = None,
    state_groups: Optional[List[List[int]]] = None,
    group_ylabels: Optional[List[str]] = None,
    show_col_labels: bool = True,
    references: Optional[List] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Plot closed-loop MPC result(s).

    Parameters
    ----------
    result
        ``MPCResult`` or list of ``MPCResult``.
        Each result has ``.t``, ``.x``, ``.u``, ``.cost``, ``.metadata``.
    state_names
        Labels for state variables.
    control_names
        Labels for control variables.
    target
        Setpoint target shown as a dashed horizontal line.
        * ``float`` — applied to the first state
        * ``dict``  — ``{"T_R": 136.0}`` matched against ``state_names``
        * ``list``  — per-state targets (use ``None`` to skip a state)
        * ``None``  — no target lines
    labels
        Column titles for multi-case plots.
    state_groups
        Group multiple states onto one shared row.
        E.g. ``[[0, 1], [2, 3]]`` plots states 0&1 together on row 0
        and states 2&3 together on row 1.  Each state in a group gets
        a distinct color from ``_STATE_COLORS``.
        ``None`` (default) uses one row per state.
    group_ylabels
        Custom y-axis labels for each group row when ``state_groups``
        is given.  Auto-generated from ``state_names`` when ``None``.
    show_col_labels
        If ``False``, suppress per-column titles (IC labels).
    save_path
        If given, save figure here.
    show
        If True, call ``plt.show()``.
    title
        Figure suptitle.  Pass ``""`` to suppress entirely.
        ``None`` (default) auto-generates from metadata.

    Returns
    -------
    (fig, axes)
    """
    from matplotlib.lines import Line2D

    # Normalise to list
    if not isinstance(result, (list, tuple)):
        results = [result]
    else:
        results = list(result)

    n_cases = len(results)

    r0 = results[0]
    n_state   = r0.x.shape[1]
    u_arr0    = np.asarray(r0.u, dtype=float)
    n_control = u_arr0.shape[1] if u_arr0.ndim == 2 else 1

    # Defaults for labels
    if state_names is None:
        state_names = [f"state {i}" for i in range(n_state)]
    if control_names is None:
        control_names = [f"u_{i}" for i in range(n_control)]
    col_labels = list(labels) if labels else [
        f"IC {i+1}: {list(np.round(results[i].x[0], 2))}" for i in range(n_cases)
    ]

    # Resolve target per-state list
    target_per_state = _resolve_mpc_target(target, state_names)

    # Resolve grouping
    _groups = [list(g) for g in state_groups] if state_groups is not None else None
    n_rows_state = len(_groups) if _groups is not None else n_state
    n_rows = n_rows_state + n_control

    row_h = 3.2 if _groups is not None else 2.6
    fig, axes = plt.subplots(
        n_rows, n_cases,
        figsize=(4.5 * n_cases, row_h * n_rows + 0.5),
        squeeze=False,
        sharex=True,
    )

    for col, res in enumerate(results):
        t_arr = np.asarray(res.t, dtype=float)   # already in display units (min for CSTR)
        x_arr = np.asarray(res.x, dtype=float)   # (n_steps+1, n_state)
        u_arr = np.asarray(res.u, dtype=float)   # (n_steps,) or (n_steps, n_control)
        if u_arr.ndim == 1:
            u_arr = u_arr[:, np.newaxis]

        if _groups is not None:
            # Grouped state rows
            ref_c = (np.asarray(references[col], dtype=float)
                     if references is not None and col < len(references) else None)
            has_ref = ref_c is not None

            for grp_idx, grp_indices in enumerate(_groups):
                ax = axes[grp_idx, col]
                grp_targets = []   # (state_name_full, tgt_value)
                for i_in_grp, s_idx in enumerate(grp_indices):
                    color = _STATE_COLORS[s_idx % len(_STATE_COLORS)]
                    short = (state_names[s_idx].split("[")[0].strip()
                             if s_idx < len(state_names) else f"state {s_idx}")
                    ax.plot(t_arr, x_arr[:, s_idx],
                            lw=1.8, color=color, label=short, zorder=3)
                    if has_ref:
                        ref_lbl = "reference" if i_in_grp == 0 else "_nolegend_"
                        ax.plot(t_arr, ref_c[:, s_idx],
                                color="k", lw=1.0, ls="--", zorder=4, label=ref_lbl)
                    tgt_i = target_per_state[s_idx] if s_idx < len(target_per_state) else None
                    if tgt_i is not None:
                        sname = (state_names[s_idx] if s_idx < len(state_names)
                                 else f"state {s_idx}")
                        grp_targets.append((sname, tgt_i))

                # Draw target lines on TOP of state lines (zorder=5)
                for sname, tgt_i in grp_targets:
                    tgt_short = sname.split("[")[0].strip()
                    ax.axhline(tgt_i, color=_C_TGT, ls="--", lw=1.3, zorder=5,
                               label=f"{tgt_short} = {tgt_i:.0f}°C")

                _tight_x(ax, t_arr)
                if col == 0:
                    if group_ylabels and grp_idx < len(group_ylabels):
                        ax.set_ylabel(group_ylabels[grp_idx])
                    else:
                        ax.set_ylabel(_group_ylabel_with_unit(grp_indices, state_names))
                if grp_idx == 0 and show_col_labels:
                    ax.set_title(col_labels[col], pad=4)
        else:
            # Per-state rows (original behaviour)
            for row in range(n_state):
                ax  = axes[row, col]
                lbl = state_names[row] if row < len(state_names) else f"state {row}"
                ax.plot(t_arr, x_arr[:, row], lw=2.0, color="C0")

                tgt_i = target_per_state[row] if row < len(target_per_state) else None
                if tgt_i is not None:
                    ax.axhline(tgt_i, color=_C_TGT, ls="--", lw=1.3, zorder=5,
                               label=f"target={tgt_i}")
                    ax.legend(fontsize=8, loc="best")

                _tight_x(ax, t_arr)
                if col == 0:
                    ax.set_ylabel(lbl)
                if row == 0 and show_col_labels:
                    ax.set_title(col_labels[col], pad=4)

        # Control rows
        ctrl_offset = n_rows_state
        for c_idx in range(n_control):
            row = ctrl_offset + c_idx
            ax  = axes[row, col]
            u_i = u_arr[:, c_idx]
            ax.step(t_arr[:-1], u_i, where="post", color=_C_CTRL, lw=1.8)
            if col == 0:
                ax.set_ylabel(
                    control_names[c_idx] if c_idx < len(control_names) else f"u_{c_idx}")
            ax.set_xlabel("$t$ [min]")
            _tight_x(ax, t_arr[:-1])

    # Unify y-limits across columns for each grouped state row
    if _groups is not None:
        for grp_idx in range(len(_groups)):
            y_min = min(axes[grp_idx, c].get_ylim()[0] for c in range(n_cases))
            y_max = max(axes[grp_idx, c].get_ylim()[1] for c in range(n_cases))
            for c in range(n_cases):
                axes[grp_idx, c].set_ylim(y_min, y_max)

    # Suptitle: non-empty title → show; "" → suppress; None → auto-generate
    if title:
        fig.suptitle(title, fontsize=13)
    elif title is None:
        meta = results[0].metadata or {}
        mode  = meta.get("mode", "")
        sys_n = meta.get("system_name", "")
        n_steps = meta.get("n_steps", len(results[0].t) - 1)
        fig.suptitle(f"{sys_n} MPC  ({mode}, {n_steps} steps)", fontsize=13)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


# ── 4. plot_inverse_params ───────────────────────────────────────────────────

def plot_inverse_params(
    param_history: Dict[str, List[float]],
    true_params:      Optional[Dict[str, float]] = None,
    estimated_params: Optional[Dict[str, float]] = None,
    save_path: Optional[str]  = None,
    show:      bool           = False,
    title:     Optional[str]  = None,
    figsize:   Optional[Tuple[float, float]] = None,
) -> Tuple[Any, Any]:
    """Parameter-convergence plot for inverse training results.

    Mirrors the style of Fig. 9 in the ADA paper:
    all identified parameters are overlaid on a single axis, each with
    a color-matched dashed horizontal line for the true value.

    Parameters
    ----------
    param_history
        ``{name: [v_step0, v_step1, ...]}`` — recorded during training.
    true_params
        Optional ``{name: true_value}`` — drawn as horizontal dashed lines.
    estimated_params
        Optional ``{name: final_value}`` — appended to legend labels.
    save_path
        File path (PNG/PDF) to save.
    show
        Call ``plt.show()`` after drawing.
    title
        Axes title.  Auto-generated when ``None``.

    Returns
    -------
    (fig, ax)
    """
    from matplotlib.lines import Line2D

    names = [n for n, h in param_history.items() if len(h) > 0]
    if not names:
        raise ValueError("plot_inverse_params: param_history contains no data.")

    n_steps = max(len(param_history[n]) for n in names)
    steps   = np.arange(n_steps)

    scale   = 1e4
    x_label = r"# Iterations ($\times 10^4$)"
    x_plot  = steps / scale

    _PAPER_RC = {
        "font.family":           "serif",
        "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":      "custom",
        "mathtext.rm":           "Times New Roman",
        "mathtext.it":           "Times New Roman:italic",
        "font.size":             16,
        "axes.labelsize":        16,
        "xtick.labelsize":       14,
        "ytick.labelsize":       14,
        "lines.linewidth":       1.8,
        "axes.grid":             False,
        "axes.spines.top":       True,
        "axes.spines.right":     True,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "xtick.top":             False,
        "ytick.right":           False,
        "savefig.bbox":          "tight",
        "savefig.dpi":           400,
    }

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=figsize or (5, 4))

        for idx, name in enumerate(names):
            color = _STATE_COLORS[idx % len(_STATE_COLORS)]
            hist  = np.asarray(param_history[name], dtype=float)
            x_h   = np.arange(len(hist)) / scale

            ax.plot(x_h, hist, color=color, lw=1.8, zorder=3)

            if true_params is not None and name in true_params:
                tv = float(true_params[name])
                ax.axhline(tv, color=color, ls="--", lw=1.5, zorder=2)

        _tight_x(ax, x_plot)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Parameter value")
        if title:
            ax.set_title(title)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()

    return fig, ax


# ── 5. plot_operator_inference ────────────────────────────────────────────────

def plot_operator_inference(
    cases: List[Dict],
    state_names: Optional[List[str]] = None,
    control_names: Optional[List[str]] = None,
    t_unit: str = "s",
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Multi-panel operator inference validation plot.

    Layout: ``(n_state + n_control)`` rows × ``n_cases`` columns.
    Reference trajectories in solid black, operator predictions in red dashed,
    control sequences as indigo step plots — matching the 'Operator ADA for MPC'
    slide style.

    Parameters
    ----------
    cases
        List of dicts, each with:
        - ``"t"``     : ndarray ``(T,)`` — time axis
        - ``"y_op"``  : ndarray ``(T, n_state)`` — operator rollout
        - ``"y_ref"`` : ndarray ``(T, n_state)`` or ``None`` — reference
        - ``"u"``     : ndarray ``(T-1,)`` or ``(T-1, n_control)`` or ``None``
    state_names
        Labels for each state row.
    control_names
        Labels for each control row.
    t_unit
        X-axis unit label (default ``"s"``).
    save_path
        If given, save figure here.
    show
        If ``True``, call ``plt.show()``.
    title
        Figure suptitle.

    Returns
    -------
    (fig, axes)
    """
    if not cases:
        raise ValueError("cases is empty")

    n_cases = len(cases)
    c0      = cases[0]

    y_op0   = np.asarray(c0["y_op"], dtype=float)
    n_state = y_op0.shape[1] if y_op0.ndim == 2 else 1

    u0       = c0.get("u")
    if u0 is not None:
        u0 = np.asarray(u0, dtype=float)
        n_control = u0.shape[1] if u0.ndim == 2 else 1
    else:
        n_control = 0

    if state_names is None:
        state_names = [f"state {i}" for i in range(n_state)]
    if control_names is None:
        control_names = [f"$u_{{{i}}}$" for i in range(n_control)]

    n_rows  = n_state + n_control
    col_w   = 3.0
    row_h   = 2.3
    fig, axes = plt.subplots(
        n_rows, n_cases,
        figsize=(col_w * n_cases, row_h * n_rows),
        squeeze=False,
    )

    for col, case in enumerate(cases):
        t_ax  = np.asarray(case["t"],    dtype=float)
        y_op  = _ensure_time_state(np.asarray(case["y_op"], dtype=float), len(t_ax))
        y_ref = case.get("y_ref")
        if y_ref is not None:
            y_ref = _ensure_time_state(np.asarray(y_ref, dtype=float), len(t_ax))

        u = case.get("u")
        if u is not None:
            u = np.asarray(u, dtype=float)
            if u.ndim == 1:
                u = u[:, np.newaxis]

        # ── state rows ──────────────────────────────────────────────────
        for row in range(n_state):
            ax = axes[row, col]

            if y_ref is not None:
                ax.plot(t_ax, y_ref[:, row], _C_REF,   lw=1.8, label="reference")
            ax.plot(t_ax, y_op[:, row],
                    color=_C_PRED, ls="--", lw=1.5, label="operator")

            ax.set_xlim(t_ax[0], t_ax[-1])
            ylim = _safe_ylim(y_ref[:, row] if y_ref is not None else y_op[:, row])
            if ylim:
                ax.set_ylim(*ylim)

            if col == 0:
                lbl = state_names[row] if row < len(state_names) else f"state {row}"
                ax.set_ylabel(lbl, fontsize=9)
                if row == 0 and y_ref is not None:
                    ax.legend(fontsize=7, loc="best")
            if row == 0:
                ax.set_title(f"Case {col + 1}", fontsize=9)
            ax.tick_params(labelbottom=(row == n_state - 1 and n_control == 0))

        # ── control rows ─────────────────────────────────────────────────
        for c_idx in range(n_control):
            row = n_state + c_idx
            ax  = axes[row, col]

            if u is not None and c_idx < u.shape[1]:
                t_ctrl = t_ax[:len(u)]
                ax.step(t_ctrl, u[:, c_idx], where="post", color=_C_CTRL, lw=1.5)
                ax.axhline(0.0, color="k", lw=0.6, ls="--")
                ax.set_xlim(t_ax[0], t_ax[-1])

            if col == 0:
                clbl = control_names[c_idx] if c_idx < len(control_names) else f"u_{c_idx}"
                ax.set_ylabel(clbl, fontsize=9)
            ax.set_xlabel(f"$t$ [{t_unit}]", fontsize=9)

    fig.suptitle(title or "Operator NN — Inference Validation",
                 fontsize=10, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def _resolve_mpc_target(
    target: Any,
    state_names: List[str],
) -> List[Optional[float]]:
    """Convert target spec to per-state list of floats (or None)."""
    n = len(state_names)
    if target is None:
        return [None] * n
    if isinstance(target, (int, float)):
        return [float(target)] + [None] * (n - 1)
    if isinstance(target, dict):
        result = [None] * n
        for i, name in enumerate(state_names):
            # strip all LaTeX markup: "$T_R$ [°C]" → "T_R"
            key = name.replace("$", "").split("[")[0].strip()
            if key in target:
                result[i] = float(target[key])
            elif name in target:
                result[i] = float(target[name])
        return result
    if isinstance(target, (list, tuple)):
        out = []
        for v in target:
            out.append(float(v) if v is not None else None)
        while len(out) < n:
            out.append(None)
        return out
    return [None] * n
