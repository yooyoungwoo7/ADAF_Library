#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_mpc_cstr.py  — CSTR MPC inference
실행:
  $env:PROBLEM = "cstr_mpc"
  $env:BASIS   = "lpa"
  python main_mpc_cstr.py
"""
from __future__ import annotations
import argparse, glob, re, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

os.environ.setdefault("PROBLEM", "cstr_mpc")
os.environ.setdefault("BASIS",   "lpa")

from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0
from problems.cstr_mpc_problem import Q_RANGE
from problems.registry import get_problem
from models.learner import OperatorLearner
from data.dataset_builder import load_segments

X_MIN = np.array([0.0, 0.0,  50.0,  50.0], dtype=np.float32)
X_MAX = np.array([5.0, 5.0, 200.0, 200.0], dtype=np.float32)

# 5개의 초기 조건
IC_LIST = [
    np.array([0.8,  0.5, 141.0, 141.0], dtype=np.float32),  # T_R >> T_ref
    np.array([1.5,  0.9, 138.5, 136.0], dtype=np.float32),  # T_R 약간 위
    np.array([1.2,  0.7, 134.0, 131.0], dtype=np.float32),  # T_R < T_ref
    np.array([0.4,  0.2, 125.0, 120.0], dtype=np.float32),  # T_R 크게 낮음
    np.array([1.8,  1.3, 136.5, 135.0], dtype=np.float32),  # T_R ≈ T_ref (농도 다름)
]


def find_latest_weight(results_root="results/cstr_mpc"):
    pattern = os.path.join(results_root, "*", "checkpoints", "*best*.weights.h5")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No best checkpoint under {results_root}")
    return files[0]


def predict_next(learner, xk, Q_val):
    z = np.concatenate([xk, [float(Q_val)]]).astype(np.float32)[None, :]
    out = learner.predict_segment(z)
    return np.clip(out["x_end"][0].astype(np.float32), X_MIN, X_MAX)


def seg_cost(Q_val, learner, xk, T_ref):
    xn = predict_next(learner, xk, float(Q_val))
    return float((xn[2] - T_ref) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  type=str,   default=None)
    parser.add_argument("--n_steps",  type=int,   default=20)
    parser.add_argument("--T_ref",    type=float, default=136.0,  # 도달 가능한 값
                        help="Target T_R [°C]. steady-state 분석: Q=-8500→134.9°C, Q=-6000→136.8°C")
    parser.add_argument("--out_dir",  type=str,   default="results/mpc_cstr")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── normalization stats ────────────────────────────────
    from config import get_data_paths
    paths = get_data_paths("cstr_mpc")
    train_seg = load_segments(paths["train_segment"])
    x_mean = train_seg.get("X_mean")
    x_std  = train_seg.get("X_std")
    print(f"[INFO] x_mean = {x_mean}")
    print(f"[INFO] x_std  = {x_std}")

    # ── Load learner ───────────────────────────────────────
    problem = get_problem("cstr_mpc")
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

    T_seg_min = (T_FINAL - T0) / N_SEG * 60.0
    print(f"[INFO] weights : {weight_path}")
    print(f"[INFO] T_seg   : {T_seg_min:.2f} min  N_SEG={N_SEG}")
    print(f"[INFO] T_ref   : {args.T_ref} °C")

    # ── 3개 IC에 대해 MPC 루프 ─────────────────────────────
    t_arr        = np.arange(args.n_steps + 1) * T_seg_min   # [min]
    state_labels = ["$C_A$ [mol/l]", "$C_B$ [mol/l]", "$T_R$ [°C]", "$T_K$ [°C]"]
    all_results  = []

    for ic_idx, x0 in enumerate(IC_LIST):
        print(f"\n{'='*55}")
        print(f"[IC {ic_idx+1}] x0 = {x0}")
        xk    = x0.copy()
        x_log = [xk.copy()]
        Q_log = []
        J_log = []

        t_ic_start = time.perf_counter()
        for step in range(args.n_steps):
            res = minimize_scalar(
                fun    = lambda Q: seg_cost(Q, learner, xk, args.T_ref),
                bounds = Q_RANGE,
                method = "bounded",
                options= {"xatol": 0.1, "maxiter": 500},
            )
            Q_apply = float(np.clip(res.x, Q_RANGE[0], Q_RANGE[1]))
            xk = predict_next(learner, xk, Q_apply)

            x_log.append(xk.copy())
            Q_log.append(Q_apply)
            J_log.append(float(res.fun))

            print(f"  [step {step+1:3d}]"
                  f"  Q*={Q_apply:8.0f} kJ/h"
                  f"  T_R={float(xk[2]):.2f}°C"
                  f"  J={res.fun:.3e}")

        elapsed = time.perf_counter() - t_ic_start
        print(f"  [IC {ic_idx+1}] total time: {elapsed:.3f} s  "
              f"({elapsed/args.n_steps*1000:.1f} ms/step)")

        all_results.append(dict(
            x0      = x0,
            x_arr   = np.stack(x_log, axis=0),
            Q_log   = np.array(Q_log),
            J_log   = np.array(J_log),
            elapsed = elapsed,
        ))

    # ── Timing summary table ───────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'IC':>3}  {'C_A':>5}  {'C_B':>5}  {'T_R':>6}  {'T_K':>6}"
          f"  {'Total (s)':>10}  {'ms/step':>8}  {'Final T_R (°C)':>14}")
    print(f"{'─'*72}")
    for ic_idx, r in enumerate(all_results):
        x0      = r["x0"]
        elapsed = r["elapsed"]
        t_r_final = float(r["x_arr"][-1, 2])
        print(f"  {ic_idx+1:>3}  {x0[0]:>5.2f}  {x0[1]:>5.2f}"
              f"  {x0[2]:>6.1f}  {x0[3]:>6.1f}"
              f"  {elapsed:>10.3f}  {elapsed/args.n_steps*1000:>8.1f}"
              f"  {t_r_final:>14.2f}")
    print(f"{'─'*72}\n")

    # ── Grid plot: rows=states+Q, cols=ICs ────────────────
    n_col = len(IC_LIST)
    n_row = 5                              # C_A, C_B, T_R, T_K, Q̇
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(4.5 * n_col, 2.8 * n_row),
        sharex=True,
    )

    for col, r in enumerate(all_results):
        x0  = r["x0"]
        xa  = r["x_arr"]       # (n_steps+1, 4)
        Ql  = r["Q_log"]       # (n_steps,)

        axes[0, col].set_title(
            f"IC {col+1}\n"
            f"$C_A$={x0[0]:.2f}, $C_B$={x0[1]:.2f}\n"
            f"$T_R$={x0[2]:.1f}°C, $T_K$={x0[3]:.1f}°C",
            fontsize=9,
        )

        # state rows
        for row, lbl in enumerate(state_labels):
            ax = axes[row, col]
            ax.plot(t_arr, xa[:, row], lw=2, color="C0")
            if row == 2:                   # T_R에만 T_ref 기준선
                ax.axhline(args.T_ref, color="red", ls="--", lw=1.3,
                           label=f"$T_{{ref}}$={args.T_ref}°C")
                ax.legend(fontsize=8, loc="best")
            if col == 0:
                ax.set_ylabel(lbl, fontsize=9)
            ax.grid(True, alpha=0.3)

        # Q̇ row
        ax = axes[n_row - 1, col]
        ax.step(t_arr[:-1], Ql, where="post", color="#534AB7", lw=1.8)
        if col == 0:
            ax.set_ylabel("$\\dot{Q}$ [kJ/h]", fontsize=9)
        ax.set_xlabel("$t$ [min]", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CSTR MPC — {len(IC_LIST)} Initial Conditions  "
        f"(N_SEG={N_SEG}, n_steps={args.n_steps}, "
        f"$T_{{ref}}$={args.T_ref}°C)",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "mpc_result.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n[DONE] plot → {plot_path}")


if __name__ == "__main__":
    main()
