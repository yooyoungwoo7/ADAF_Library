#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_mpc_bioreactor.py — Fed-batch bioreactor economic MPC
Economic objective: maximize terminal product amount Ps * Vs [g]

실행:
  $env:PROBLEM = "bioreactor"
  $env:BASIS   = "lpa"
  python main_mpc_bioreactor.py
"""
from __future__ import annotations

import argparse, glob, re, os, time, csv
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerBase

class _MultiSegHandler(HandlerBase):
    """Legend handle rendered as N equal-width coloured segments."""
    def __init__(self, colors, lw=2.2):
        self._colors = colors
        self._lw = lw
        super().__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        n = len(self._colors)
        artists = []
        for i, c in enumerate(self._colors):
            x0 = xdescent + i * width / n
            x1 = xdescent + (i + 1) * width / n
            line = plt.Line2D([x0, x1], [ydescent + height / 2] * 2,
                              lw=self._lw, color=c, transform=trans)
            artists.append(line)
        return artists

_MPC_COLORS = ["C5", "C2", "C1", "C3"]   # Q1 Q2 Q3 Q4 순서

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.8,
    "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "mathtext.fontset": "stix",
}

os.environ.setdefault("PROBLEM", "bioreactor")
os.environ.setdefault("BASIS",   "lpa")

from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0, get_data_paths
from problems.registry import get_problem
from models.learner import OperatorLearner
from data.dataset_builder import load_segments

# ── Problem constants ────────────────────────────────────────────
INP_MIN   = 0.005    # L/min — lower bound on feed rate
INP_MAX   = 0.200    # L/min — upper bound on feed rate
YX_FIXED  = 0.40     # biomass yield (fixed; within training range 0.3–0.5)
SIN_FIXED = 0.8      # feed substrate concentration [g/L] (within 0.3–1.5)
T_SEG_MIN = (T_FINAL - T0) / N_SEG   # segment duration [min]

V_MAX_DEFAULT = 5.0  # L — max reactor volume (matches training upper bound)
S_INH         = 0.8  # g/L — substrate inhibition soft-constraint threshold (≈ Sin_max)

# State clipping — Ss bounded by Sin_max ≈ 1.5 g/L
X_MIN = np.array([  0.0,  0.0,  0.0,  0.0], dtype=np.float32)
X_MAX = np.array([ 20.0,  5.0, 20.0, 15.0], dtype=np.float32)

# Default initial condition for fed-batch start (do-mpc reference: Xs=1, Ss=0.3)
IC_DEFAULT = np.array([1.0, 0.3, 0.0, 1.0], dtype=np.float32)
# Xs=1.0 g/L, Ss=0.3 g/L, Ps=0.0 g/L, Vs=1.0 L

STATE_LABELS = ["$X_s$ [g/L]", "$S_s$ [g/L]", "$P_s$ [g/L]", "$V_s$ [L]"]
STATE_COLORS = ["C0", "C1", "C2", "C4"]
BASELINE_LS  = ["--", "-.", ":"]


# ── Utilities ────────────────────────────────────────────────────

def find_latest_weight(results_root="results/bioreactor"):
    pattern = os.path.join(results_root, "*", "checkpoints", "*best*.weights.h5")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No best checkpoint under {results_root}")
    return files[0]


# ── Operator-based prediction ────────────────────────────────────

def predict_next(learner, xk, inp, Yx=YX_FIXED, Sin=SIN_FIXED):
    """Single-segment operator prediction.
    Input: [Xs, Ss, Ps, Vs, Yx, Sin, inp] (7D).  Returns clipped next state.
    """
    z = np.array([xk[0], xk[1], xk[2], xk[3],
                  float(Yx), float(Sin), float(inp)],
                 dtype=np.float32)[None, :]
    out = learner.predict_segment(z)
    return np.clip(out["x_end"][0].astype(np.float32), X_MIN, X_MAX)


def rollout_horizon(learner, xk, inp_seq, Yx=YX_FIXED, Sin=SIN_FIXED):
    """Multi-step operator rollout over a control sequence.
    Returns states array of shape (len(inp_seq)+1, 4) including x0.
    """
    states = [xk.copy()]
    for inp in inp_seq:
        xk = predict_next(learner, xk,
                          float(np.clip(inp, INP_MIN, INP_MAX)), Yx, Sin)
        states.append(xk)
    return np.stack(states, axis=0)


# ── True ODE plant ───────────────────────────────────────────────

def plant_step(problem, xk, inp, Yx=YX_FIXED, Sin=SIN_FIXED):
    """Advance the TRUE ODE plant by one segment (T_SEG_MIN min) via RK45."""
    theta = np.array([Yx, Sin, float(inp)], dtype=np.float64)
    sol = solve_ivp(
        fun    = lambda t, y: problem.rhs_np(t, y, theta),
        t_span = (0.0, T_SEG_MIN),
        y0     = np.asarray(xk, dtype=np.float64),
        method = "RK45",
        rtol   = 1e-6,
        atol   = 1e-8,
    )
    return np.clip(sol.y[:, -1].astype(np.float32), X_MIN, X_MAX)


# ── Economic objective ───────────────────────────────────────────

def economic_cost(inp_flat, learner, xk, inp_prev, Yx, Sin,
                  w_product, w_stage, w_feed, w_smooth, w_volume, w_substrate,
                  w_nonneg, w_ss_track, Ss_opt, V_max):
    """Economic MPC objective (minimized by scipy):

    J = -w_product  * Ps_N * Vs_N                        (terminal product amount)
        - w_stage   * sum(Ps_k * Vs_k) * T_seg           (running product accumulation)
        + w_feed    * sum(inp_k) * T_seg                  (total feed volume)
        + w_smooth  * sum((inp_k - inp_{k-1})^2)          (move suppression)
        + w_volume  * sum(max(0, Vs_k - V_max)^2)         (soft volume constraint)
        + w_substrate * sum(max(0, Ss_k - S_inh)^2)       (substrate inhibition)
        + w_nonneg  * sum(max(0, -state_k)^2)             (non-negativity)
        + w_ss_track* sum((Ss_k - Ss_opt)^2)              (Ss tracking toward optimal μ)
    """
    inp_seq = np.asarray(inp_flat, dtype=np.float64)
    states  = rollout_horizon(learner, xk, inp_seq, Yx, Sin)   # (N+1, 4)

    # Terminal product amount Ps_N * Vs_N [g]
    J_product   = -w_product * float(states[-1, 2]) * float(states[-1, 3])

    # Running (stage) cost: reward Ps*Vs at every predicted step (like do-mpc lterm)
    PsVs_traj   = states[1:, 2].astype(np.float64) * states[1:, 3].astype(np.float64)
    J_stage     = -w_stage * float(np.sum(PsVs_traj)) * T_SEG_MIN

    # Feed cost (proportional to total feed volume added)
    J_feed      = w_feed * float(np.sum(inp_seq)) * T_SEG_MIN

    # Move suppression
    inp_ext     = np.concatenate([[inp_prev], inp_seq])
    J_smooth    = w_smooth * float(np.sum((inp_ext[1:] - inp_ext[:-1]) ** 2))

    # Soft constraint: reactor volume
    Vs_arr      = states[1:, 3].astype(np.float64)
    J_volume    = w_volume * float(np.sum(np.maximum(0.0, Vs_arr - V_max) ** 2))

    # Soft constraint: substrate inhibition
    Ss_arr      = states[1:, 1].astype(np.float64)
    J_substrate = w_substrate * float(np.sum(np.maximum(0.0, Ss_arr - S_INH) ** 2))

    # Non-negativity
    J_nonneg    = w_nonneg * float(np.sum(np.maximum(0.0, -states[1:].astype(np.float64)) ** 2))

    # Ss tracking: drive Ss toward Haldane optimal μ point (√(K_M·K_I) ≈ 0.5 g/L)
    J_ss_track  = w_ss_track * float(np.sum((Ss_arr - Ss_opt) ** 2))

    return J_product + J_stage + J_feed + J_smooth + J_volume + J_substrate + J_nonneg + J_ss_track


# ── Economic MPC closed-loop ─────────────────────────────────────

def run_mpc(learner, problem, x0, args):
    """Shrinking-horizon economic MPC.
    Prediction model: operator.  Plant model: true ODE (RK45).
    """
    n_steps  = args.n_steps
    N_pred   = args.n_pred
    xk       = x0.astype(np.float32).copy()
    inp_prev = float(args.inp_init)

    x_log   = [xk.copy()]
    inp_log = []
    J_log   = []
    t_log   = []

    for step in range(n_steps):
        N_h      = min(N_pred, n_steps - step)   # shrinking horizon
        bounds_h = [(INP_MIN, INP_MAX)] * N_h
        u0       = np.full(N_h, inp_prev, dtype=np.float64)

        def obj(u):
            return economic_cost(
                u, learner, xk, inp_prev,
                args.Yx, args.Sin,
                args.w_product, args.w_stage, args.w_feed, args.w_smooth,
                args.w_volume, args.w_substrate, args.w_nonneg,
                args.w_ss_track, args.Ss_opt,
                args.V_max,
            )

        # Multi-start: warm-start + high-feed start → pick best
        t_start = time.perf_counter()
        candidates = [u0, np.full(N_h, INP_MAX, dtype=np.float64)]
        best_res = None
        for u_cand in candidates:
            r = minimize(obj, u_cand, method="SLSQP", bounds=bounds_h,
                         options={"ftol": args.ftol, "maxiter": args.maxiter})
            if best_res is None or r.fun < best_res.fun:
                best_res = r
        res = best_res
        t_ms = (time.perf_counter() - t_start) * 1000.0

        inp_apply = float(np.clip(res.x[0], INP_MIN, INP_MAX))

        # Advance the TRUE plant
        xk = plant_step(problem, xk, inp_apply, args.Yx, args.Sin)

        x_log.append(xk.copy())
        inp_log.append(inp_apply)
        J_log.append(float(res.fun))
        t_log.append(t_ms)
        inp_prev = inp_apply

        PsVs = float(xk[2] * xk[3])
        print(f"  [{step+1:3d}/{n_steps}]"
              f"  inp={inp_apply:.4f} L/min"
              f"  Xs={xk[0]:.3f}  Ss={xk[1]:.3f}"
              f"  Ps={xk[2]:.3f}  Vs={xk[3]:.3f}"
              f"  PsVs={PsVs:.3f} g"
              f"  {t_ms:.1f} ms")

    return dict(
        x_arr   = np.stack(x_log, axis=0),   # (n_steps+1, 4)
        inp_arr = np.array(inp_log),          # (n_steps,)
        J_arr   = np.array(J_log),
        t_solve = np.array(t_log),            # [ms]
    )


# ── Constant-feed baselines ──────────────────────────────────────

def run_baseline(problem, x0, inp_const, n_steps, Yx=YX_FIXED, Sin=SIN_FIXED):
    """Simulate with constant feed rate using the true ODE plant."""
    xk    = x0.astype(np.float32).copy()
    x_log = [xk.copy()]
    for _ in range(n_steps):
        xk = plant_step(problem, xk, inp_const, Yx, Sin)
        x_log.append(xk.copy())
    x_arr = np.stack(x_log, axis=0)
    return dict(
        x_arr         = x_arr,
        inp_arr       = np.full(n_steps, inp_const),
        product_final = float(x_arr[-1, 2] * x_arr[-1, 3]),
    )


# ── Plotting ─────────────────────────────────────────────────────

def make_plots(out_dir, ts, t_arr, mpc_r, baselines, args):
    with plt.rc_context(_PAPER_RC):
        _make_plots_inner(out_dir, ts, t_arr, mpc_r, baselines, args)


def _make_plots_inner(out_dir, ts, t_arr, mpc_r, baselines, args):
    # Drop const-high for cleaner comparison
    baselines_plot = {k: v for k, v in baselines.items() if k != "high"}
    t_ctrl = t_arr[:-1]

    # ── Main 1×4 figure ──────────────────────────────────────────
    cum_feed_mpc = np.concatenate([[0.0], np.cumsum(mpc_r["inp_arr"] * T_SEG_MIN)])
    PsVs_mpc     = mpc_r["x_arr"][:, 2] * mpc_r["x_arr"][:, 3]
    eff_mpc      = np.where(cum_feed_mpc > 1e-9, PsVs_mpc / cum_feed_mpc, 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharex=True)

    # P1: control input
    ax = axes[0]
    ax.step(t_ctrl, mpc_r["inp_arr"], where="post", lw=2.2,
            color="C5")
    bl_handles = []
    for k, (name, br) in enumerate(baselines_plot.items()):
        h, = ax.plot([], [], lw=1.3, ls=BASELINE_LS[k], color="gray", alpha=0.75)
        ax.axhline(br["inp_arr"][0], lw=1.3,
                   ls=BASELINE_LS[k], color="gray", alpha=0.75)
        bl_handles.append((h, f"const {name}"))
    ax.set_ylabel("$u_{inp}$ [L/min]")
    ax.set_ylim(INP_MIN - 0.005, INP_MAX + 0.005)
    ax.set_xlabel("$t$ [min]")

    proxy_mpc = plt.Line2D([], [], lw=0)
    handles = [proxy_mpc] + [h for h, _ in bl_handles]
    labels  = ["Economic MPC"] + [lbl for _, lbl in bl_handles]
    ax.legend(handles, labels,
              handler_map={proxy_mpc: _MultiSegHandler(_MPC_COLORS)},
              frameon=False, loc="upper right")

    # P2: product amount
    ax = axes[1]
    ax.plot(t_arr, PsVs_mpc, lw=2.2, color="C2")
    for k, (name, br) in enumerate(baselines_plot.items()):
        PsVs_b = br["x_arr"][:, 2] * br["x_arr"][:, 3]
        ax.plot(t_arr, PsVs_b, lw=1.3,
                ls=BASELINE_LS[k], color="gray", alpha=0.75)
    ax.set_ylabel("Product amount [g]")
    ax.set_xlabel("$t$ [min]")

    # P3: cumulative feed volume
    ax = axes[2]
    ax.plot(t_arr, cum_feed_mpc, lw=2.2, color="C1")
    for k, (name, br) in enumerate(baselines_plot.items()):
        cum_feed_b = np.concatenate([[0.0], np.cumsum(br["inp_arr"] * T_SEG_MIN)])
        ax.plot(t_arr, cum_feed_b, lw=1.3,
                ls=BASELINE_LS[k], color="gray", alpha=0.75)
    ax.set_ylabel("Cumulative feed volume [L]")
    ax.set_xlabel("$t$ [min]")

    # P4: feed efficiency
    ax = axes[3]
    ax.plot(t_arr, eff_mpc, lw=2.2, color="C3")
    for k, (name, br) in enumerate(baselines_plot.items()):
        cum_feed_b = np.concatenate([[0.0], np.cumsum(br["inp_arr"] * T_SEG_MIN)])
        eff_b = np.where(cum_feed_b > 1e-9,
                         br["x_arr"][:, 2] * br["x_arr"][:, 3] / cum_feed_b,
                         0.0)
        ax.plot(t_arr, eff_b, lw=1.3,
                ls=BASELINE_LS[k], color="gray", alpha=0.75)
    ax.set_ylabel("Feed efficiency [g/L]")
    ax.set_xlabel("$t$ [min]")

    for ax in axes:
        ax.set_xlim(t_arr[0], t_arr[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"states_{ts}.png"))
    plt.close(fig)

    # ── Control input (standalone) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step(t_ctrl, mpc_r["inp_arr"], where="post", lw=2.2,
            color="C1", label="Economic MPC")
    for k, (name, br) in enumerate(baselines_plot.items()):
        ax.step(t_ctrl, br["inp_arr"], where="post", lw=1.3,
                ls=BASELINE_LS[k], color="gray", alpha=0.75,
                label=f"const {name}")
    ax.set_xlabel("$t$ [min]")
    ax.set_ylabel("$u_{inp}$ [L/min]")
    ax.set_ylim(INP_MIN - 0.005, INP_MAX + 0.005)
    ax.set_xlim(t_arr[0], t_arr[-1])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"control_{ts}.png"))
    plt.close(fig)

    # ── MPC solve time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(np.arange(1, len(mpc_r["t_solve"]) + 1), mpc_r["t_solve"],
           color="C3", width=0.7)
    mean_ms = mpc_r["t_solve"].mean()
    ax.axhline(mean_ms, color="k", ls="--", lw=1.2,
               label=f"mean={mean_ms:.1f} ms")
    ax.set_xlabel("MPC step")
    ax.set_ylabel("Solve time [ms]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"solve_time_{ts}.png"))
    plt.close(fig)


# ── CSV saving ───────────────────────────────────────────────────

def save_csv(out_dir, ts, t_arr, mpc_r, baselines):
    # Closed-loop states
    with open(os.path.join(out_dir, f"closed_loop_states_{ts}.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_min", "Xs_g_L", "Ss_g_L", "Ps_g_L", "Vs_L",
                    "PsVs_g"])
        for i, t in enumerate(t_arr):
            row = mpc_r["x_arr"][i]
            w.writerow([f"{t:.4f}",
                        f"{row[0]:.6f}", f"{row[1]:.6f}",
                        f"{row[2]:.6f}", f"{row[3]:.6f}",
                        f"{row[2]*row[3]:.6f}"])

    # Closed-loop controls
    with open(os.path.join(out_dir, f"closed_loop_controls_{ts}.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_min", "inp_L_per_min", "solve_ms"])
        for i, (t, inp, tms) in enumerate(
                zip(t_arr[:-1], mpc_r["inp_arr"], mpc_r["t_solve"])):
            w.writerow([f"{t:.4f}", f"{inp:.6f}", f"{tms:.2f}"])

    # Summary metrics
    T_span = T_FINAL - T0
    product_mpc  = float(mpc_r["x_arr"][-1, 2] * mpc_r["x_arr"][-1, 3])
    feed_mpc     = float(np.sum(mpc_r["inp_arr"]) * T_SEG_MIN)
    rows = [["economic_mpc",
             f"{product_mpc:.4f}",
             f"{feed_mpc:.4f}",
             f"{product_mpc/T_span:.6f}",
             f"{mpc_r['t_solve'].mean():.2f}",
             f"{mpc_r['t_solve'].max():.2f}"]]
    for name, br in baselines.items():
        feed_b = float(np.sum(br["inp_arr"]) * T_SEG_MIN)
        rows.append([f"const_{name}",
                     f"{br['product_final']:.4f}",
                     f"{feed_b:.4f}",
                     f"{br['product_final']/T_span:.6f}",
                     "N/A", "N/A"])

    with open(os.path.join(out_dir, f"summary_metrics_{ts}.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "product_g", "total_feed_L",
                    "productivity_g_per_min",
                    "mean_solve_ms", "max_solve_ms"])
        w.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fed-batch bioreactor economic MPC")
    parser.add_argument("--weights",      type=str,   default=None)
    parser.add_argument("--n_steps",      type=int,   default=N_SEG,
                        help=f"Total MPC steps (default N_SEG={N_SEG})")
    parser.add_argument("--n_pred",       type=int,   default=10,
                        help="Prediction horizon [segments, default 10 = 20 min]")
    parser.add_argument("--inp_init",     type=float, default=0.04,
                        help="Warm-start feed rate [L/min]")
    parser.add_argument("--Yx",           type=float, default=YX_FIXED,
                        help="Fixed biomass yield")
    parser.add_argument("--Sin",          type=float, default=SIN_FIXED,
                        help="Fixed feed substrate concentration [g/L]")
    # Economic weights
    parser.add_argument("--w_product",    type=float, default=1.0,
                        help="Weight on terminal Ps*Vs")
    parser.add_argument("--w_stage",      type=float, default=0.1,
                        help="Weight on running (stage) Ps*Vs sum — like do-mpc lterm")
    parser.add_argument("--w_feed",       type=float, default=0.01,
                        help="Weight on total feed volume")
    parser.add_argument("--w_smooth",     type=float, default=0.1,
                        help="Move-suppression weight")
    parser.add_argument("--w_volume",     type=float, default=10.0,
                        help="Soft penalty weight on Vs > V_max")
    parser.add_argument("--w_substrate",  type=float, default=0.5,
                        help="Soft penalty on Ss > S_inh")
    parser.add_argument("--w_nonneg",     type=float, default=10.0,
                        help="Non-negativity penalty weight")
    parser.add_argument("--w_ss_track",   type=float, default=0.0,
                        help="Weight on (Ss - Ss_opt)^2 tracking term")
    parser.add_argument("--Ss_opt",       type=float, default=0.5,
                        help="Optimal substrate for max mu: sqrt(K_M*K_I)=0.5 g/L")
    parser.add_argument("--V_max",        type=float, default=V_MAX_DEFAULT,
                        help=f"Max reactor volume [L] (default {V_MAX_DEFAULT})")
    parser.add_argument("--maxiter",      type=int,   default=150,
                        help="SLSQP max iterations per start (default 150)")
    parser.add_argument("--ftol",         type=float, default=1e-5,
                        help="SLSQP convergence tolerance (default 1e-5)")
    # Initial condition
    parser.add_argument("--Xs0", type=float, default=float(IC_DEFAULT[0]),
                        help="Initial biomass [g/L]")
    parser.add_argument("--Ss0", type=float, default=float(IC_DEFAULT[1]),
                        help="Initial substrate [g/L]")
    parser.add_argument("--Ps0", type=float, default=float(IC_DEFAULT[2]),
                        help="Initial product [g/L]")
    parser.add_argument("--Vs0", type=float, default=float(IC_DEFAULT[3]),
                        help="Initial volume [L]")
    parser.add_argument("--out_dir",      type=str,
                        default="results/bioreactor_economic_mpc")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load normalization stats ─────────────────────────────────
    paths     = get_data_paths("bioreactor")
    train_seg = load_segments(paths["train_segment"])
    x_mean    = train_seg.get("X_mean")
    x_std     = train_seg.get("X_std")
    print(f"[INFO] x_mean = {x_mean}")
    print(f"[INFO] x_std  = {x_std}")

    # ── Load operator ────────────────────────────────────────────
    problem = get_problem("bioreactor")
    learner = OperatorLearner(
        problem  = problem,
        hidden   = HIDDEN,
        n_layers = N_LAYERS,
        lr       = 1e-3,
        x_mean   = x_mean,
        x_std    = x_std,
    )
    weight_path = args.weights or find_latest_weight()
    learner.load_weights(weight_path)
    print(f"[INFO] weights : {weight_path}")
    print(f"[INFO] T_seg   : {T_SEG_MIN:.2f} min  N_SEG={N_SEG}")
    print(f"[INFO] n_steps : {args.n_steps}  n_pred={args.n_pred}")
    print(f"[INFO] Yx={args.Yx}  Sin={args.Sin}  V_max={args.V_max}")

    t_arr = np.arange(args.n_steps + 1) * T_SEG_MIN   # time axis [min]
    x0    = np.array([args.Xs0, args.Ss0, args.Ps0, args.Vs0], dtype=np.float32)
    print(f"\n[IC] Xs={x0[0]}  Ss={x0[1]}  Ps={x0[2]}  Vs={x0[3]} L")

    # ── Economic MPC ─────────────────────────────────────────────
    print(f"\n{'='*60}\n  Economic MPC\n{'='*60}")
    mpc_result = run_mpc(learner, problem, x0, args)

    # ── Constant-feed baselines ───────────────────────────────────
    baselines = {}
    for name, inp_val in [("low", 0.005), ("nominal", 0.04), ("high", 0.20)]:
        print(f"\n[Baseline] const-{name}  inp={inp_val} L/min")
        baselines[name] = run_baseline(
            problem, x0, inp_val, args.n_steps, args.Yx, args.Sin)

    # ── Save plots & CSVs ─────────────────────────────────────────
    make_plots(args.out_dir, ts, t_arr, mpc_result, baselines, args)
    save_csv(args.out_dir, ts, t_arr, mpc_result, baselines)

    for fname in [f"states_{ts}.png", f"control_{ts}.png",
                  f"product_amount_{ts}.png", f"solve_time_{ts}.png",
                  f"economic_terms_{ts}.png"]:
        print(f"[DONE] plot → {os.path.join(args.out_dir, fname)}")
    print(f"[DONE] csv  → {args.out_dir}/closed_loop_*.csv  summary_metrics_*.csv")

    # ── Summary table ─────────────────────────────────────────────
    T_span       = T_FINAL - T0
    product_mpc  = float(mpc_result["x_arr"][-1, 2] * mpc_result["x_arr"][-1, 3])
    feed_mpc     = float(np.sum(mpc_result["inp_arr"]) * T_SEG_MIN)
    t_mean       = mpc_result["t_solve"].mean()
    t_max        = mpc_result["t_solve"].max()

    print(f"\n{'─'*72}")
    print(f"  {'Strategy':<20}  {'Product [g]':>11}  {'Feed [L]':>8}  "
          f"{'g/min':>8}  {'mean ms':>8}  {'max ms':>7}")
    print(f"{'─'*72}")
    print(f"  {'Economic MPC':<20}  {product_mpc:>11.4f}  {feed_mpc:>8.4f}  "
          f"  {product_mpc/T_span:>8.5f}  {t_mean:>8.1f}  {t_max:>7.1f}")
    for name, br in baselines.items():
        feed_b = float(np.sum(br["inp_arr"]) * T_SEG_MIN)
        print(f"  {f'const {name}':<20}  {br['product_final']:>11.4f}  "
              f"{feed_b:>8.4f}    {br['product_final']/T_span:>8.5f}"
              f"  {'—':>8}  {'—':>7}")
    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
