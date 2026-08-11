#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_operator_quality.py  —  operator 예측 품질 확인
몇 개 케이스에 대해 reference ODE trajectory와 비교 플롯.

실행:
  $env:PROBLEM = "cstr_mpc"
  $env:BASIS   = "lpa"
  python check_operator_quality.py [--n_plot 4] [--split test] [--seed 42]
"""
from __future__ import annotations
import argparse, glob, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# 어느 디렉토리에서 실행해도 data_files/, results/ 경로가 맞도록
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PROBLEM", "cstr_mpc")
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


def rollout_q_seq(learner, x0, Q_seq):
    """cstr_mpc 전용 rollout.
    세그먼트 k마다 Q_seq[k]를 theta로 사용해 전진적분.
    반환: {"t": (NT_TOTAL,), "x": (NT_TOTAL,4), "residual": (NT_TOTAL,4)}
    """
    problem = learner.problem
    basis   = learner.basis
    xk = np.asarray(x0, dtype=np.float32).copy()
    t_segs, x_segs, res_segs = [], [], []
    t_offset = 0.0

    for k, Q_k in enumerate(Q_seq):
        # 세그먼트 입력: [x_k (4), Q_k (1)]
        Xk = np.concatenate([xk, [float(Q_k)]]).astype(np.float32)[None, :]
        out = learner.predict_segment(Xk)       # batch=1

        x_seg    = out["x"]                     # (1, NT_SEG, 4)
        xdot_seg = out["xdot"][0]               # (NT_SEG, 4)

        # physics residual: ẋ_pred − f(x_pred, Q_k)
        theta_k = tf.constant([[float(Q_k)]], dtype=DTYPE)   # (1,1)
        rhs_k   = problem.rhs_tf(
            tf.constant(x_seg, dtype=DTYPE),    # (1, NT_SEG, 4)
            theta_k,
        ).numpy()[0]                            # (NT_SEG, 4)
        res = xdot_seg - rhs_k                  # (NT_SEG, 4)

        x_seg = x_seg[0]                        # (NT_SEG, 4)
        t_local = basis.t_local_np + t_offset
        if k > 0:                               # 이전 segment 끝점 중복 제거
            t_local = t_local[1:]
            x_seg   = x_seg[1:]
            res     = res[1:]

        t_segs.append(t_local)
        x_segs.append(x_seg)
        res_segs.append(res)
        t_offset += basis.T_seg
        xk = out["x_end"][0]                    # (4,)

    return {
        "t":        np.concatenate(t_segs),      # (NT_TOTAL,)
        "x":        np.concatenate(x_segs),      # (NT_TOTAL, 4)
        "residual": np.concatenate(res_segs),    # (NT_TOTAL, 4)
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--n_plot",  type=int, default=4,
                        help="플롯할 케이스 수")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/mpc_cstr")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── normalization stats ────────────────────────────────
    paths     = get_data_paths("cstr_mpc")
    train_seg = load_segments(paths["train_segment"])
    x_mean    = train_seg.get("X_mean")
    x_std     = train_seg.get("X_std")

    # ── load model ─────────────────────────────────────────
    problem = get_problem("cstr_mpc")
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
    X0      = fc["X0"].astype(np.float32)        # (n_cases, 4+N_SEG)
    Y_ref   = fc["Y_ref"].astype(np.float32)     # (n_cases, NT_TOTAL, 4)
    t_grid  = fc["t_grid"].astype(np.float32)    # (NT_TOTAL,)
    n_cases = X0.shape[0]
    print(f"[INFO] {args.split}: {n_cases} cases, Y_ref={Y_ref.shape}")

    # ── sample cases ───────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(n_cases, size=min(args.n_plot, n_cases), replace=False))

    # ── rollout & metrics ──────────────────────────────────
    state_labels = problem.state_plot_labels()   # ("$C_A$", "$C_B$", "$T_R$", "$T_K$")
    time_factor  = problem.time_factor            # 60 → 시간 [h] → 분 [min]
    t_min        = t_grid * time_factor
    T_seg_min    = (T_FINAL - T0) / N_SEG * time_factor   # 1.2 min
    seg_t        = np.arange(N_SEG) * T_seg_min            # 각 세그먼트 시작 시각 [min]

    results = []
    print(f"\n{'case':>6}  {'L2_tot':>10}  {'L2_CA':>10}  {'L2_CB':>10}  {'L2_TR':>10}  {'L2_TK':>10}")
    print("─" * 66)
    for i in idx:
        x0_i  = X0[i, :problem.state_dim]        # (4,)
        Q_seq = X0[i, problem.state_dim:]         # (N_SEG,)
        x_ref = Y_ref[i]                          # (NT_TOTAL, 4)

        out      = rollout_q_seq(learner, x0_i, Q_seq)
        x_pred   = out["x"]                       # (NT_TOTAL, 4)
        res_norm = np.linalg.norm(out["residual"], axis=-1)  # (NT_TOTAL,)

        sw_l2  = statewise_l2_rel(x_pred, x_ref)
        tot_l2 = l2_rel(x_pred.ravel(), x_ref.ravel())
        results.append(dict(
            i=i, x0=x0_i, x_pred=x_pred, x_ref=x_ref,
            residual=out["residual"], res_norm=res_norm,
            Q_seq=Q_seq, sw_l2=sw_l2, tot_l2=tot_l2,
        ))
        print(f"{i:6d}  {tot_l2:10.2e}  " +
              "  ".join(f"{e:10.2e}" for e in sw_l2))

    # ── figure: cols=cases, rows=(states + Q_seq) ─────────
    n_col = len(results)
    n_row = len(state_labels) + 1     # 4 states + Q_seq
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(3.5 * n_col, 2.4 * n_row),
        sharex=True,
    )
    if n_col == 1:
        axes = axes[:, None]

    for col, r in enumerate(results):
        axes[0, col].set_title(
            f"case {r['i']}\n"
            f"Q: [{r['Q_seq'].min():.0f}, {r['Q_seq'].max():.0f}]\n"
            f"L2={r['tot_l2']:.2e}",
            fontsize=8,
        )
        # state rows
        for row, lab in enumerate(state_labels):
            ax = axes[row, col]
            ax.plot(t_min, r["x_ref"][:, row],
                    color="k", lw=1.8, label="reference" if col == 0 else "_")
            ax.plot(t_min, r["x_pred"][:, row],
                    "--", color="C3", lw=1.5, label="operator" if col == 0 else "_")
            if col == 0:
                ax.set_ylabel(lab, fontsize=9)
                ax.legend(fontsize=7, loc="best", handlelength=1.2)
            tight_x(ax, t_min)

        # Q_seq row
        ax = axes[n_row - 1, col]
        ax.step(seg_t, r["Q_seq"], where="post", color="#534AB7", lw=1.5)
        if col == 0:
            ax.set_ylabel(r"$\dot{Q}$ [kJ/h]", fontsize=9)
        ax.set_xlabel("$t$ [min]", fontsize=8)
        ax.set_xlim(t_min[0], t_min[-1])

    mean_l2 = np.mean([r["tot_l2"] for r in results])
    fig.suptitle(
        f"Operator quality  ({args.split}, N={n_col})   mean L2={mean_l2:.2e}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "operator_quality.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[DONE] → {out_path}")

    # ── summary ────────────────────────────────────────────
    all_l2 = np.array([r["tot_l2"] for r in results])
    all_sw  = np.stack([r["sw_l2"]  for r in results])
    print(f"\n{'─'*46}")
    print(f"  {'state':6s}  {'mean L2':>10s}  {'max L2':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    print(f"  {'total':6s}  {all_l2.mean():10.2e}  {all_l2.max():10.2e}")
    for j, lab in enumerate(["CA", "CB", "TR", "TK"]):
        print(f"  {lab:6s}  {all_sw[:,j].mean():10.2e}  {all_sw[:,j].max():10.2e}")
    print(f"{'─'*46}\n")

    # ── per-case inference plots ───────────────────────────
    infer_dir   = os.path.join(args.out_dir, "inference_cases")
    os.makedirs(infer_dir, exist_ok=True)
    state_names = problem.state_labels          # ("C_A", "C_B", "T_R", "T_K")

    # Q_seq를 시간축으로 변환: 세그먼트 k → t_start [min]
    T_seg_min  = (T_FINAL - T0) / N_SEG * problem.time_factor
    seg_t      = np.arange(N_SEG) * T_seg_min   # (N_SEG,) 각 세그먼트 시작 시각 [min]

    for r in results:
        n_rows = len(state_labels) + 1           # 4 states + Q_seq
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows + 0.5), sharex=False)

        # ── title & subtitle ──────────────────────────────
        x0 = r["x0"]
        fig.suptitle(
            f"Inference  case {r['i']}  (L2={r['tot_l2']:.2e})\n"
            f"$C_A$={x0[0]:.2f}  $C_B$={x0[1]:.2f}  "
            f"$T_R$={x0[2]:.1f}  $T_K$={x0[3]:.1f}",
            fontsize=11,
        )

        # ── state rows ────────────────────────────────────
        for row, lab in enumerate(state_labels):
            ax = axes[row]
            ax.plot(t_min, r["x_ref"][:, row],  color="k",   lw=1.8, label="reference")
            ax.plot(t_min, r["x_pred"][:, row], "--", color="C3", lw=1.5, label="operator")
            ax.set_ylabel(lab)
            ax.legend(fontsize=8, loc="best", handlelength=1.2)
            tight_x(ax, t_min)

        # ── Q_seq row (step plot, 세그먼트별 입력) ─────────
        ax_q = axes[-1]
        ax_q.step(seg_t, r["Q_seq"], where="post", color="#534AB7", lw=1.8)
        ax_q.set_ylabel(r"$\dot{Q}$ [kJ/h]")
        ax_q.set_xlabel("$t$ [min]")
        ax_q.set_xlim(seg_t[0], seg_t[-1] + T_seg_min)
        ax_q.axhline(0, color="k", lw=0.5, ls="--")

        fig.tight_layout()
        save_path = os.path.join(infer_dir, f"case_{r['i']:03d}_traj.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── residual profile (별도 파일) ──────────────────
        plot_residual_profile(
            t         = t_grid,
            residual  = r["residual"],
            labels    = [f"res_{n}" for n in state_names],
            save_path = os.path.join(infer_dir, f"case_{r['i']:03d}_residual.png"),
            time_unit = problem.time_unit,
            time_factor = problem.time_factor,
        )
        print(f"[INFER] case {r['i']:03d} → {infer_dir}")

    print(f"[DONE] inference plots → {infer_dir}/")


if __name__ == "__main__":
    main()
