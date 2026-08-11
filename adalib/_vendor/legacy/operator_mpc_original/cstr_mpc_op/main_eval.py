#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_eval.py
============================================================
Validation / inference entry point.
  - validation mode: pick N cases from val_fullcase.npz, rollout, plot, report metrics.
  - custom mode: rollout one user-provided x_input (input_dim floats).

Rollout here is INFERENCE-ONLY (no gradient) — chains N_SEG segments by
feeding the previous segment's x_end as the next segment's IC.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

# Apply --problem / --basis early. If --basis omitted, we'll auto-detect from
# the run's config_snapshot.json after CLI parsing (still before learner build).
def _early_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--basis", default=None)
    args, _ = parser.parse_known_args()
    if args.problem is not None:
        os.environ["PROBLEM"] = args.problem.lower()
    if args.basis is not None:
        os.environ["BASIS"] = args.basis.lower()


_early_overrides()

if os.environ.get("USE_GPU", "0") != "1":
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

from config import (
    PROBLEM_NAME, N_SEG, INPUT_DIM, get_data_paths, get_result_dir,
)
from problems.registry import get_problem
from data.dataset_builder import load_segments, load_fullcase
from models.learner import OperatorLearner
from utils.io_utils import ensure_dir, save_json, save_csv_rows
from utils.metrics import l2_rel, statewise_l2_rel
from utils.plotting import plot_state_vs_reference, plot_residual_profile
from utils.sweep_plots import plot_param_sweeps, plot_val_comparison, plot_random_cases


def _find_snapshot(weight_path):
    d = os.path.dirname(os.path.abspath(weight_path))
    for _ in range(4):
        cand = os.path.join(d, "config_snapshot.json")
        if os.path.exists(cand):
            return cand
        d = os.path.dirname(d)
    return None


def _build_learner(problem, weight_path, hidden=None, n_layers=None, basis_name=None):
    snap_basis = None
    if hidden is None or n_layers is None or basis_name is None:
        snap_path = _find_snapshot(weight_path)
        if snap_path is not None:
            with open(snap_path) as f:
                snap = json.load(f)
            if hidden is None:
                hidden = int(snap.get("hidden", 256))
            if n_layers is None:
                n_layers = int(snap.get("n_layers", 3))
            snap_basis = snap.get("basis")
            print(f"[cfg] loaded arch from {snap_path}: hidden={hidden}, "
                  f"n_layers={n_layers}, basis={snap_basis}")
        else:
            hidden = hidden or 256
            n_layers = n_layers or 3

    basis_to_use = basis_name or snap_basis  # CLI > snapshot > BASIS_NAME default

    train = load_segments(get_data_paths()["train_segment"])
    learner = OperatorLearner(
        problem=problem,
        hidden=hidden,
        n_layers=n_layers,
        x_mean=train.get("X_mean"),
        x_std=train.get("X_std"),
        basis_name=basis_to_use,
    )
    learner.load_weights(weight_path)
    return learner


def _describe(problem, x_input):
    x0 = x_input[:problem.state_dim]
    theta = x_input[problem.state_dim:problem.state_dim + problem.param_dim]
    st = ", ".join(f"{n}={v:.4f}" for n, v in zip(problem.state_labels, x0))
    pt = ", ".join(f"{n}={v:.4f}" for n, v in zip(problem.param_labels, theta))
    return f"IC = ({st})   |   params = ({pt})"


def _rollout_and_report(learner, x_input, out_dir, tag):
    problem = learner.problem
    rollout = learner.rollout_full_trajectory(x_input, n_seg=N_SEG)
    t = rollout["t"]; pred = rollout["x"]; residual = rollout["residual"]
    x0 = x_input[:problem.state_dim]
    theta = x_input[problem.state_dim:problem.state_dim + problem.param_dim]
    ref = problem.solve_reference(theta=theta, x0=x0, t_grid=t)

    plot_state_vs_reference(
        t, pred, ref, labels=problem.state_plot_labels(),
        title_prefix=tag,
        save_path=os.path.join(out_dir, f"{tag}_traj.png"),
        subtitle=_describe(problem, x_input),
        time_unit=problem.time_unit,
        extras=problem.extra_plot_traces() or None,
        theta=theta,
        time_factor=getattr(problem, "time_factor", 1.0),
    )
    plot_residual_profile(
        t, residual, labels=tuple(f"res_{lab}" for lab in problem.state_labels),
        save_path=os.path.join(out_dir, f"{tag}_residual.png"),
        time_unit=problem.time_unit,
        time_factor=getattr(problem, "time_factor", 1.0),
    )
    np.savez_compressed(
        os.path.join(out_dir, f"{tag}_rollout.npz"),
        x_input=x_input, t=t, x_pred=pred, x_ref=ref, residual=residual,
    )
    return {
        "l2_rel_total": l2_rel(pred, ref),
        "l2_rel_states": statewise_l2_rel(pred, ref).tolist(),
    }


def _run_validation(learner, fullcase_path, out_dir, n_cases,
                    comparison_n=4, sweep_n=8, random_n=4, random_seed=None):
    problem = learner.problem
    full = load_fullcase(fullcase_path)
    X0 = np.asarray(full["X0"], dtype=np.float32)
    n_cases = min(int(n_cases), X0.shape[0])

    rows = [["case_idx", "l2_rel_total"] + [f"l2_{lab}" for lab in problem.state_labels]]
    per_case = []
    for i in range(n_cases):
        tag = f"case_{i:03d}"
        stats = _rollout_and_report(learner, X0[i], out_dir, tag)
        rows.append([i, stats["l2_rel_total"]] + stats["l2_rel_states"])
        per_case.append(stats)
        print(f"[VAL] {tag}  L2_rel={stats['l2_rel_total']:.3e}  per-state={stats['l2_rel_states']}")

    save_csv_rows(rows[1:], os.path.join(out_dir, "validation_metrics.csv"), header=rows[0])
    save_json({"per_case": per_case}, os.path.join(out_dir, "validation_summary.json"))

    if comparison_n > 0:
        cmp_path = os.path.join(out_dir, "comparison.png")
        plot_val_comparison(learner, fullcase_path, cmp_path, n_cases=comparison_n)
        print(f"[SAVE] comparison plot -> {cmp_path}")
    if sweep_n > 0:
        sweep_path = os.path.join(out_dir, "param_sweeps.png")
        plot_param_sweeps(learner, sweep_path, n=sweep_n)
        print(f"[SAVE] param sweeps -> {sweep_path}")
    if random_n > 0:
        rand_path = os.path.join(out_dir, "random_cases.png")
        plot_random_cases(learner, rand_path, n_cases=random_n, seed=random_seed)
        print(f"[SAVE] random cases -> {rand_path}")


def main():
    parser = argparse.ArgumentParser(description="Operator library — eval / inference entry point.")
    parser.add_argument("--problem", default=None)
    parser.add_argument("--basis", default=None,
                        help="Override basis (else auto-loaded from snapshot).")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", choices=["validation", "custom"], default="validation")
    parser.add_argument("--fullcase", default=None)
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--x_input", type=float, nargs="*", default=None,
                        help=f"{INPUT_DIM} floats for custom mode")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--comparison_n", type=int, default=4)
    parser.add_argument("--sweep_n", type=int, default=8)
    parser.add_argument("--random_n", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=None)
    args = parser.parse_args()

    problem = get_problem(PROBLEM_NAME)
    out_dir = args.out_dir or os.path.join(
        get_result_dir(), f"{problem.name}_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ensure_dir(out_dir)

    learner = _build_learner(problem, args.weights,
                             hidden=args.hidden, n_layers=args.n_layers,
                             basis_name=args.basis)

    if args.mode == "validation":
        fullcase = args.fullcase or get_data_paths()["val_fullcase"]
        _run_validation(learner, fullcase, out_dir, args.n_cases,
                        comparison_n=args.comparison_n, sweep_n=args.sweep_n,
                        random_n=args.random_n, random_seed=args.random_seed)
        print(f"[SAVE] validation report -> {out_dir}")
        return

    expected = problem.input_dim
    if args.x_input is None or len(args.x_input) != expected:
        raise ValueError(f"Custom mode for {problem.name} needs x_input with length {expected}.")
    x_input = np.asarray(args.x_input, dtype=np.float32)
    stats = _rollout_and_report(learner, x_input, out_dir, tag="custom_input")
    save_json(stats, os.path.join(out_dir, "custom_input_metrics.json"))
    print(f"[SAVE] custom inference -> {out_dir}  L2_rel={stats['l2_rel_total']:.3e}")


if __name__ == "__main__":
    main()
