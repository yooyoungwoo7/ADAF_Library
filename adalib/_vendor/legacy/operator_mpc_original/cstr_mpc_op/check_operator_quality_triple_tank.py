#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_operator_quality_triple_tank.py  —  Triple Tank operator 예측 품질 확인
몇 개 케이스에 대해 reference ODE trajectory와 비교 플롯.

실행:
  $env:PROBLEM = "triple_tank_mpc"
  $env:BASIS   = "lpa"
  python check_operator_quality_triple_tank.py [--n_plot 4] [--split test] [--seed 42]
"""
from __future__ import annotations
import argparse, glob, re, os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PROBLEM", "triple_tank_mpc")
os.environ.setdefault("BASIS",   "lpa")

from utils._style import apply_style, tight_x
apply_style()

from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0, get_data_paths, TF_DTYPE
from problems.registry import get_problem
from models.learner import OperatorLearner
from data.dataset_builder import load_segments, load_fullcase
from utils.metrics import statewise_l2_rel, l2_rel
from utils.plotting import plot_residual_profile

DTYPE = getattr(tf, TF_DTYPE)


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def find_latest_weight(results_root="results/triple_tank_mpc"):
    pattern = os.path.join(results_root, "*", "checkpoints", "*best*.weights.h5")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No best checkpoint under {results_root}")
    return files[0]


def rollout_q_seq(learner, x0, theta):
    """triple_tank_mpc 전용 rollout.
    theta: (N_SEG*2,) — [Q1_0,...,Q1_{N-1}, Q2_0,...,Q2_{N-1}]
    세그먼트 k마다 (Q1_seq[k], Q2_seq[k])를 입력으로 사용해 전진적분.
    반환: {"t": (NT_TOTAL,), "x": (NT_TOTAL,3), "residual": (NT_TOTAL,3)}
    """
    problem = learner.problem
    basis   = learner.basis
    Q1_seq  = np.asarray(theta[:N_SEG], dtype=np.float32)
    Q2_seq  = np.asarray(theta[N_SEG:], dtype=np.float32)
    xk = np.asarray(x0, dtype=np.float32).copy()
    t_segs, x_segs, res_segs = [], [], []
    t_offset = 0.0

    for k in range(N_SEG):
        # 세그먼트 입력: [h1_k, h2_k, h3_k, Q1_k, Q2_k]
        Xk = np.array([xk[0], xk[1], xk[2],
                       float(Q1_seq[k]), float(Q2_seq[k])],
                      dtype=np.float32)[None, :]
        out = learner.predict_segment(Xk)

        x_seg    = out["x"]           # (1, NT_SEG, 3)
        xdot_seg = out["xdot"][0]     # (NT_SEG, 3)

        # physics residual: ẋ_pred − f(x_pred, Q1_k, Q2_k)
        theta_k = tf.constant([[float(Q1_seq[k]), float(Q2_seq[k])]], dtype=DTYPE)
        rhs_k   = problem.rhs_tf(
            tf.constant(x_seg, dtype=DTYPE),
            theta_k,
        ).numpy()[0]                  # (NT_SEG, 3)
        res = xdot_seg - rhs_k

        x_seg   = x_seg[0]            # (NT_SEG, 3)
        t_local = basis.t_local_np + t_offset
        if k > 0:
            t_local = t_local[1:]
            x_seg   = x_seg[1:]
            res     = res[1:]

        t_segs.append(t_local)
        x_segs.append(x_seg)
        res_segs.append(res)
        t_offset += basis.T_seg
        xk = out["x_end"][0]

    return {
        "t":        np.concatenate(t_segs),
        "x":        np.concatenate(x_segs),
        "residual": np.concatenate(res_segs),
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--n_plot",  type=int, default=4)
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/mpc_triple_tank")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── normalization stats ────────────────────────────────
    paths     = get_data_paths("triple_tank_mpc")
    train_seg = load_segments(paths["train_segment"])
    x_mean    = train_seg.get("X_mean")
    x_std     = train_seg.get("X_std")

    # ── load model ─────────────────────────────────────────
    problem = get_problem("triple_tank_mpc")
    learner = OperatorLearner(
        problem=problem, hidden=HIDDEN, n_layers=N_LAYERS,
        lr=1e-3, x_mean=x_mean, x_std=x_std,
    )
    weight_path = args.weights or find_latest_weight()
    learner.load_weights(weight_path)
    print(f"[INFO] weights : {weight_path}")

    # ── load fullcase ──────────────────────────────────────
    fc_path = paths[f"{args.split}_fullcase"]
    fc      = load_fullcase(fc_path)
    X0      = fc["X0"].astype(np.float32)      # (n_cases, 3 + N_SEG*2)
    Y_ref   = fc["Y_ref"].astype(np.float32)   # (n_cases, NT_TOTAL, 3)
    t_grid  = fc["t_grid"].astype(np.float32)  # (NT_TOTAL,)
    n_cases = X0.shape[0]
    print(f"[INFO] {args.split}: {n_cases} cases, Y_ref={Y_ref.shape}")

    # ── sample cases ───────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(n_cases, size=min(args.n_plot, n_cases), replace=False))

    # ── rollout & metrics ──────────────────────────────────
    state_labels = problem.state_plot_labels()
    T_seg_s   = (T_FINAL - T0) / N_SEG
    seg_t     = np.arange(N_SEG) * T_seg_s    # 각 세그먼트 시작 [s]

    results = []
    print(f"\n{'case':>6}  {'L2_tot':>10}  {'L2_h1':>10}  {'L2_h2':>10}  {'L2_h3':>10}")
    print("─" * 56)
    for i in idx:
        x0_i   = X0[i, :problem.state_dim]    # (3,)
        theta_i = X0[i, problem.state_dim:]    # (N_SEG*2,)
        x_ref  = Y_ref[i]                      # (NT_TOTAL, 3)

        out      = rollout_q_seq(learner, x0_i, theta_i)
        x_pred   = out["x"]
        res_norm = np.linalg.norm(out["residual"], axis=-1)

        sw_l2  = statewise_l2_rel(x_pred, x_ref)
        tot_l2 = l2_rel(x_pred.ravel(), x_ref.ravel())
        results.append(dict(
            i=i, x0=x0_i, x_pred=x_pred, x_ref=x_ref,
            residual=out["residual"], res_norm=res_norm,
            theta=theta_i, sw_l2=sw_l2, tot_l2=tot_l2,
        ))
        print(f"{i:6d}  {tot_l2:10.2e}  " +
              "  ".join(f"{e:10.2e}" for e in sw_l2))

    # ── operator_quality.png ───────────────────────────────
    n_col = len(results)
    n_row = len(state_labels) + 1   # 3 states + Q1/Q2 step plot (2 rows)
    n_row += 1                       # Q1 + Q2 각각 한 행씩
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(3.5 * n_col, 2.4 * n_row),
        sharex=True,
    )
    if n_col == 1:
        axes = axes[:, None]

    for col, r in enumerate(results):
        Q1_seq = r["theta"][:N_SEG]
        Q2_seq = r["theta"][N_SEG:]
        axes[0, col].set_title(
            f"case {r['i']}\n"
            f"Q1:[{Q1_seq.min():.0f},{Q1_seq.max():.0f}]  "
            f"Q2:[{Q2_seq.min():.0f},{Q2_seq.max():.0f}]\n"
            f"L2={r['tot_l2']:.2e}",
            fontsize=8,
        )
        # state rows
        for row, lab in enumerate(state_labels):
            ax = axes[row, col]
            ax.plot(t_grid, r["x_ref"][:, row],
                    color="k", lw=1.8, label="reference" if col == 0 else "_")
            ax.plot(t_grid, r["x_pred"][:, row],
                    "--", color="C3", lw=1.5, label="operator" if col == 0 else "_")
            if col == 0:
                ax.set_ylabel(lab, fontsize=9)
                ax.legend(fontsize=7, loc="best", handlelength=1.2)
            tight_x(ax, t_grid)

        # Q1 row
        ax = axes[n_row - 2, col]
        ax.step(seg_t, Q1_seq, where="post", color="C1", lw=1.5)
        if col == 0:
            ax.set_ylabel(r"$Q_1$ [cm³/s]", fontsize=9)
        ax.set_xlim(t_grid[0], t_grid[-1])

        # Q2 row
        ax = axes[n_row - 1, col]
        ax.step(seg_t, Q2_seq, where="post", color="#534AB7", lw=1.5)
        if col == 0:
            ax.set_ylabel(r"$Q_2$ [cm³/s]", fontsize=9)
        ax.set_xlabel("$t$ [s]", fontsize=8)
        ax.set_xlim(t_grid[0], t_grid[-1])

    mean_l2 = np.mean([r["tot_l2"] for r in results])
    fig.suptitle(
        f"Operator quality  ({args.split}, N={n_col})   mean L2={mean_l2:.2e}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"operator_quality_{ts}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[DONE] → {out_path}")

    # ── summary ────────────────────────────────────────────
    all_l2 = np.array([r["tot_l2"] for r in results])
    all_sw  = np.stack([r["sw_l2"]  for r in results])
    print(f"\n{'─'*42}")
    print(f"  {'state':6s}  {'mean L2':>10s}  {'max L2':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    print(f"  {'total':6s}  {all_l2.mean():10.2e}  {all_l2.max():10.2e}")
    for j, lab in enumerate(["h1", "h2", "h3"]):
        print(f"  {lab:6s}  {all_sw[:,j].mean():10.2e}  {all_sw[:,j].max():10.2e}")
    print(f"{'─'*42}\n")

    # ── per-case inference plots ───────────────────────────
    infer_dir   = os.path.join(args.out_dir, f"inference_cases_{ts}")
    os.makedirs(infer_dir, exist_ok=True)
    state_names = problem.state_labels

    for r in results:
        n_rows = len(state_labels) + 2   # 3 states + Q1 + Q2
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(10, 2.5 * n_rows + 0.5),
                                 sharex=False)
        x0 = r["x0"]
        fig.suptitle(
            f"Inference  case {r['i']}  (L2={r['tot_l2']:.2e})\n"
            f"$h_1$={x0[0]:.1f}  $h_2$={x0[1]:.1f}  $h_3$={x0[2]:.1f} cm",
            fontsize=11,
        )

        for row, lab in enumerate(state_labels):
            ax = axes[row]
            ax.plot(t_grid, r["x_ref"][:, row],  color="k",   lw=1.8, label="reference")
            ax.plot(t_grid, r["x_pred"][:, row], "--", color="C3", lw=1.5, label="operator")
            ax.set_ylabel(lab)
            ax.legend(fontsize=8, loc="best", handlelength=1.2)
            tight_x(ax, t_grid)

        Q1_seq = r["theta"][:N_SEG]
        Q2_seq = r["theta"][N_SEG:]

        ax_q1 = axes[-2]
        ax_q1.step(seg_t, Q1_seq, where="post", color="C1", lw=1.8)
        ax_q1.set_ylabel(r"$Q_1$ [cm³/s]")
        ax_q1.set_xlim(seg_t[0], seg_t[-1] + T_seg_s)

        ax_q2 = axes[-1]
        ax_q2.step(seg_t, Q2_seq, where="post", color="#534AB7", lw=1.8)
        ax_q2.set_ylabel(r"$Q_2$ [cm³/s]")
        ax_q2.set_xlabel("$t$ [s]")
        ax_q2.set_xlim(seg_t[0], seg_t[-1] + T_seg_s)

        fig.tight_layout()
        save_path = os.path.join(infer_dir, f"case_{r['i']:03d}_traj.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        plot_residual_profile(
            t           = t_grid,
            residual    = r["residual"],
            labels      = [f"res_{n}" for n in state_names],
            save_path   = os.path.join(infer_dir, f"case_{r['i']:03d}_residual.png"),
            time_unit   = problem.time_unit,
            time_factor = problem.time_factor,
        )
        print(f"[INFER] case {r['i']:03d} → {infer_dir}")

    print(f"[DONE] inference plots → {infer_dir}/")


if __name__ == "__main__":
    main()
