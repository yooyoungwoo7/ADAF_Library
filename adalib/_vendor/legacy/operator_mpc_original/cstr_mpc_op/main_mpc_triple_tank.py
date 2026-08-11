#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_mpc_triple_tank.py  — Triple Tank MPC inference
실행:
  $env:PROBLEM = "triple_tank_mpc"
  $env:BASIS   = "lpa"
  python main_mpc_triple_tank.py [--h3_target 20.0] [--n_steps 15]
"""
from __future__ import annotations
import argparse, glob, re, os, time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

os.environ.setdefault("PROBLEM", "triple_tank_mpc")
os.environ.setdefault("BASIS",   "lpa")

from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0
from problems.triple_tank_mpc_problem import Q1_RANGE, Q2_RANGE
from problems.registry import get_problem
from models.learner import OperatorLearner
from data.dataset_builder import load_segments

H_MIN = np.array([  0.0,   0.0,   0.0], dtype=np.float32)
H_MAX = np.array([400.0, 400.0, 400.0], dtype=np.float32)

# 5개 초기 조건 — PDF 참고: 의미 있는 수위 차이, h3_target=150 cm 기준
# 정상상태 (Q1=Q2=68 cm³/s): h1≈194, h2≈104, h3≈150 cm
IC_LIST = [
    np.array([140.0,  90.0,  80.0], dtype=np.float32),  # h3<목표, 자연 방향
    np.array([250.0, 200.0, 240.0], dtype=np.float32),  # 전체 높음, PDF 초기조건 유사
    np.array([ 80.0,  50.0,  40.0], dtype=np.float32),  # h3<<목표, 수위 낮음
    np.array([190.0, 120.0, 180.0], dtype=np.float32),  # h3>목표, 약간 초과
    np.array([195.0, 105.0, 150.0], dtype=np.float32),  # 정상상태 근방 (수렴 확인)
]


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


def predict_next(learner, xk, Q1, Q2):
    z = np.array([xk[0], xk[1], xk[2], float(Q1), float(Q2)], dtype=np.float32)[None, :]
    out = learner.predict_segment(z)
    return np.clip(out["x_end"][0].astype(np.float32), H_MIN, H_MAX)


def seg_cost(u, learner, xk, h3_target, h2_target, lambda_h2,
             Q1_prev, Q2_prev, reg_w=1e-3):
    Q1 = float(np.clip(u[0], Q1_RANGE[0], Q1_RANGE[1]))
    Q2 = float(np.clip(u[1], Q2_RANGE[0], Q2_RANGE[1]))
    xn = predict_next(learner, xk, Q1, Q2)
    J_h3 = (xn[2] - h3_target) ** 2
    J_h2 = (xn[1] - h2_target) ** 2
    J_reg = reg_w * ((Q1 - Q1_prev) ** 2 + (Q2 - Q2_prev) ** 2)
    return float(J_h3 + lambda_h2 * J_h2 + J_reg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   type=str,   default=None)
    parser.add_argument("--n_steps",   type=int,   default=100,
                        help="MPC 스텝 수 (N_SEG=50, T_seg=12s → 100×12=1200s)")
    parser.add_argument("--h3_target",  type=float, default=150.0,
                        help="Target h3 [cm]  (default 150 cm = PDF 정상상태)")
    parser.add_argument("--h2_target",  type=float, default=None,
                        help="Target h2 [cm]  (default: h3_target과 동일)")
    parser.add_argument("--lambda_h2",  type=float, default=1.0,
                        help="h2 추적 오차 가중치  (0이면 h3-only SISO)")
    parser.add_argument("--reg_w",      type=float, default=1e-3,
                        help="Move-suppression weight λ: J = (h3-h3_ref)² + λ·(ΔQ1²+ΔQ2²)")
    parser.add_argument("--reg_thresh", type=float, default=0.1,
                        help="J_track 이 이 값 이하일 때 reg_w 강화 (기본 0.1)")
    parser.add_argument("--reg_boost",  type=float, default=50.0,
                        help="reg_thresh 이하에서 reg_w 배율 (기본 50×)")
    parser.add_argument("--out_dir",   type=str,   default="results/mpc_triple_tank")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── normalization stats ────────────────────────────────
    from config import get_data_paths
    paths = get_data_paths("triple_tank_mpc")
    train_seg = load_segments(paths["train_segment"])
    x_mean = train_seg.get("X_mean")
    x_std  = train_seg.get("X_std")
    print(f"[INFO] x_mean = {x_mean}")
    print(f"[INFO] x_std  = {x_std}")

    # ── Load learner ───────────────────────────────────────
    problem = get_problem("triple_tank_mpc")
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

    T_seg_s = (T_FINAL - T0) / N_SEG   # [s] per segment
    print(f"[INFO] weights : {weight_path}")
    print(f"[INFO] T_seg   : {T_seg_s:.1f} s  N_SEG={N_SEG}")
    h2_target = args.h2_target if args.h2_target is not None else args.h3_target
    print(f"[INFO] h3_target: {args.h3_target} cm")
    print(f"[INFO] h2_target: {h2_target} cm  (lambda_h2={args.lambda_h2})")

    # ── 5개 IC에 대해 MPC 루프 ─────────────────────────────
    t_arr        = np.arange(args.n_steps + 1) * T_seg_s   # [s]
    state_labels = [r"$h_1$ [cm]", r"$h_2$ [cm]", r"$h_3$ [cm]"]
    all_results  = []

    for ic_idx, x0 in enumerate(IC_LIST):
        print(f"\n{'='*60}")
        print(f"[IC {ic_idx+1}] x0 = {x0}")
        xk     = x0.copy()
        x_log  = [xk.copy()]
        Q1_log = []
        Q2_log = []
        J_log  = []
        # 정상상태 근방 Q로 warm start
        Q1_prev, Q2_prev = 68.0, 68.0

        t_ic_start = time.perf_counter()
        for step in range(args.n_steps):
            # 물리 기반 초기값: h3 편차 방향에 따라 Q1, Q2 모두 결정
            if xk[2] < args.h3_target:
                # h3 낮음 → Q1/Q2 최대로 양쪽에서 채움
                x0_phys = np.array([Q1_RANGE[1], Q2_RANGE[1]])
            else:
                # h3 높음 → Q1/Q2=0으로 배수
                x0_phys = np.array([Q1_RANGE[0], Q2_RANGE[0]])

            # 멀티스타트: warm-start와 physics-start 중 best 선택
            candidates = [np.array([Q1_prev, Q2_prev]), x0_phys]
            best_res = None
            for x0_cand in candidates:
                J_track_now = ((xk[2] - args.h3_target) ** 2
                               + args.lambda_h2 * (xk[1] - h2_target) ** 2)
                reg_w_eff = (args.reg_w * args.reg_boost
                             if J_track_now < args.reg_thresh
                             else args.reg_w)
                r = minimize(
                    fun    = lambda u: seg_cost(u, learner, xk,
                                                args.h3_target, h2_target,
                                                args.lambda_h2,
                                                Q1_prev, Q2_prev, reg_w_eff),
                    x0     = x0_cand,
                    method = "Nelder-Mead",
                    options= {"xatol": 0.5, "fatol": 1e-4, "maxiter": 2000,
                              "adaptive": True},
                )
                if best_res is None or r.fun < best_res.fun:
                    best_res = r
            res = best_res
            Q1_apply = float(np.clip(res.x[0], Q1_RANGE[0], Q1_RANGE[1]))
            Q2_apply = float(np.clip(res.x[1], Q2_RANGE[0], Q2_RANGE[1]))
            xk = predict_next(learner, xk, Q1_apply, Q2_apply)

            x_log.append(xk.copy())
            Q1_log.append(Q1_apply)
            Q2_log.append(Q2_apply)
            J_log.append(float(res.fun))
            Q1_prev, Q2_prev = Q1_apply, Q2_apply   # warm-start next step

            print(f"  [step {step+1:3d}]"
                  f"  Q1*={Q1_apply:6.1f}  Q2*={Q2_apply:6.1f} cm³/s"
                  f"  h2={float(xk[1]):.2f} cm"
                  f"  h3={float(xk[2]):.2f} cm"
                  f"  J={res.fun:.3e}")

        elapsed = time.perf_counter() - t_ic_start
        print(f"  [IC {ic_idx+1}] total time: {elapsed:.3f} s  "
              f"({elapsed/args.n_steps*1000:.1f} ms/step)")

        all_results.append(dict(
            x0      = x0,
            x_arr   = np.stack(x_log, axis=0),
            Q1_log  = np.array(Q1_log),
            Q2_log  = np.array(Q2_log),
            J_log   = np.array(J_log),
            elapsed = elapsed,
        ))

    # ── Timing summary table ───────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'IC':>3}  {'h1':>5}  {'h2':>5}  {'h3':>5}"
          f"  {'Total (s)':>10}  {'ms/step':>8}  {'Final h3 (cm)':>13}")
    print(f"{'─'*72}")
    for ic_idx, r in enumerate(all_results):
        x0      = r["x0"]
        elapsed = r["elapsed"]
        h3_final = float(r["x_arr"][-1, 2])
        print(f"  {ic_idx+1:>3}  {x0[0]:>5.1f}  {x0[1]:>5.1f}  {x0[2]:>5.1f}"
              f"  {elapsed:>10.3f}  {elapsed/args.n_steps*1000:>8.1f}"
              f"  {h3_final:>13.2f}")
    print(f"{'─'*72}\n")

    # ── Grid plot: rows=[h1,h2,h3,Q1,Q2], cols=ICs ────────
    n_col = len(IC_LIST)
    n_row = 5   # h1, h2, h3, Q1, Q2
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(4.5 * n_col, 2.8 * n_row),
        sharex=True,
    )

    for col, r in enumerate(all_results):
        x0  = r["x0"]
        xa  = r["x_arr"]    # (n_steps+1, 3)
        Q1l = r["Q1_log"]   # (n_steps,)
        Q2l = r["Q2_log"]   # (n_steps,)

        axes[0, col].set_title(
            f"IC {col+1}\n"
            f"$h_1$={x0[0]:.1f}, $h_2$={x0[1]:.1f}, $h_3$={x0[2]:.1f} cm",
            fontsize=9,
        )

        # state rows
        for row, lbl in enumerate(state_labels):
            ax = axes[row, col]
            ax.plot(t_arr, xa[:, row], lw=2, color="C0")
            if row == 1:   # h2 목표 기준선
                ax.axhline(h2_target, color="red", ls="--", lw=1.3,
                           label=f"$h_{{2,ref}}$={h2_target:.1f} cm")
                ax.legend(fontsize=8, loc="best")
            if row == 2:   # h3 목표 기준선
                ax.axhline(args.h3_target, color="red", ls="--", lw=1.3,
                           label=f"$h_{{3,ref}}$={args.h3_target} cm")
                ax.legend(fontsize=8, loc="best")
            if col == 0:
                ax.set_ylabel(lbl, fontsize=9)
            ax.grid(True, alpha=0.3)

        # Q1 row
        ax = axes[3, col]
        ax.step(t_arr[:-1], Q1l, where="post", color="C1", lw=1.8)
        if col == 0:
            ax.set_ylabel("$Q_1$ [cm³/s]", fontsize=9)
        ax.set_ylim(Q1_RANGE[0] - 5, Q1_RANGE[1] + 5)
        ax.grid(True, alpha=0.3)

        # Q2 row
        ax = axes[4, col]
        ax.step(t_arr[:-1], Q2l, where="post", color="#534AB7", lw=1.8)
        if col == 0:
            ax.set_ylabel("$Q_2$ [cm³/s]", fontsize=9)
        ax.set_xlabel("$t$ [s]", fontsize=9)
        ax.set_ylim(Q2_RANGE[0] - 5, Q2_RANGE[1] + 5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Triple Tank MPC — {len(IC_LIST)} Initial Conditions  "
        f"(N_SEG={N_SEG}, n_steps={args.n_steps},  "
        f"$h_{{3,ref}}$={args.h3_target} cm,  "
        f"$h_{{2,ref}}$={h2_target:.1f} cm,  "
        f"$\\lambda_{{h2}}$={args.lambda_h2})",
        fontsize=12,
    )
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.out_dir, f"mpc_result_{ts}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[DONE] plot → {plot_path}")

    npz_path = os.path.join(args.out_dir, f"mpc_result_{ts}.npz")
    np.savez(
        npz_path,
        h_arr     = np.stack([r["x_arr"] for r in all_results]),
        Q_arr     = np.stack([np.stack([r["Q1_log"], r["Q2_log"]], axis=-1)
                               for r in all_results]),
        t_arr     = t_arr,
        ic_list   = np.stack([r["x0"]   for r in all_results]),
        h3_target = np.float32(args.h3_target),
        h2_target = np.float32(h2_target),
    )
    print(f"[DONE] npz  → {npz_path}")


if __name__ == "__main__":
    main()
