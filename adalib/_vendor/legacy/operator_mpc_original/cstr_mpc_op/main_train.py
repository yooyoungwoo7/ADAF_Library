#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_train.py
============================================================
Single training entry point for any registered problem.

Usage:
    python main_train.py                    # uses config.PROBLEM_NAME
    python main_train.py --problem cstr     # override
    PROBLEM=cstr python main_train.py       # env override

Phase-1 single-segment physics-only training is the canonical workflow.
Phase-2 K-segment chained-gradient rollout is OFF by default
(`ROLLOUT_EPOCHS=0` in config) — kept as opt-in diagnostic only.
"""

import argparse
import os
import secrets
from datetime import datetime

import numpy as np
import tensorflow as tf

# Apply --problem / --basis BEFORE importing anything that reads config.
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
        print("[device] GPU hidden (CPU+XLA). set USE_GPU=1 to expose GPU.")
    except Exception as e:
        print(f"[device] failed to hide GPU: {e}")

from config import (
    PROBLEM_NAME, BASIS_NAME,
    PHYSICS_EPOCHS, PHYSICS_BATCH_SIZE, PHYSICS_LR,
    USE_LR_SCHEDULE, LR_WARMUP_FRACTION, LR_MIN,
    ROLLOUT_EPOCHS, ROLLOUT_STEPS, ROLLOUT_ALPHA, ROLLOUT_LR,
    N_TRAIN_CASES, N_VAL_CASES, N_TEST_CASES, SEED,
    HIDDEN, N_LAYERS, LPA_N_P, MAX_ORDER, GAMMA, DT_SEG, NT_SEG, INPUT_DIM,
    ADAF_N_M, ADAF_N_P,
    get_data_paths, get_result_dir,
)
from problems.registry import get_problem
from data.dataset_builder import (
    build_and_save_fullcase, build_segments_from_fullcase,
    compute_input_stats, save_segments,
    load_segments, make_tf_dataset,
)
from models.learner import OperatorLearner
from utils.io_utils import ensure_dir, save_json
from utils.plotting import plot_training_curves


def _ensure_datasets(problem, paths, seed):
    if not all(os.path.exists(paths[k]) for k in ("train_fullcase", "val_fullcase", "test_fullcase")):
        build_and_save_fullcase(problem, paths["train_fullcase"], N_TRAIN_CASES, seed=seed)
        build_and_save_fullcase(problem, paths["val_fullcase"],   N_VAL_CASES,   seed=seed + 1)
        build_and_save_fullcase(problem, paths["test_fullcase"],  N_TEST_CASES,  seed=seed + 2)

    if not all(os.path.exists(paths[k]) for k in ("train_segment", "val_segment", "test_segment")):
        train_seg = build_segments_from_fullcase(problem, paths["train_fullcase"], seed=seed + 100)
        X_mean, X_std = compute_input_stats(train_seg["X"])
        save_segments(paths["train_segment"], train_seg, X_mean=X_mean, X_std=X_std)

        val_seg = build_segments_from_fullcase(problem, paths["val_fullcase"], seed=seed + 101)
        save_segments(paths["val_segment"], val_seg, X_mean=X_mean, X_std=X_std)

        test_seg = build_segments_from_fullcase(problem, paths["test_fullcase"], seed=seed + 102)
        save_segments(paths["test_segment"], test_seg, X_mean=X_mean, X_std=X_std)


def main():
    parser = argparse.ArgumentParser(description="Operator library — training entry point.")
    parser.add_argument("--problem", default=None,
                        help=f"Problem to load (default: {PROBLEM_NAME}). Overrides PROBLEM env.")
    parser.add_argument("--basis", default=None,
                        help=f"Basis to use: 'lpa' or 'adaf' (default: {BASIS_NAME}).")
    parser.add_argument("--epochs", type=int, default=PHYSICS_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PHYSICS_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PHYSICS_LR)
    parser.add_argument("--weights", default=None, help="Optional starting weights")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--no_lr_schedule", action="store_true")
    parser.add_argument("--warmup_frac", type=float, default=LR_WARMUP_FRACTION)
    parser.add_argument("--lr_min", type=float, default=LR_MIN)
    parser.add_argument("--rollout_epochs", type=int, default=ROLLOUT_EPOCHS,
                        help="Opt-in diagnostic K-segment chained-gradient training (default 0).")
    parser.add_argument("--rollout_steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--rollout_alpha", type=float, default=ROLLOUT_ALPHA)
    parser.add_argument("--rollout_lr", type=float, default=ROLLOUT_LR)
    parser.add_argument("--rebuild_data", action="store_true")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = secrets.randbits(31)
        print(f"[seed] generated random seed = {args.seed}")
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    problem = get_problem(PROBLEM_NAME)
    print(f"[problem] {problem.name}  state_dim={problem.state_dim} "
          f"param_dim={problem.param_dim}  cons_w={problem.cons_w}")

    paths = get_data_paths()
    if args.rebuild_data:
        for k in paths:
            if os.path.exists(paths[k]):
                os.remove(paths[k])
    _ensure_datasets(problem, paths, args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(get_result_dir(), f"{problem.name}_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    train = load_segments(paths["train_segment"])
    val   = load_segments(paths["val_segment"])
    # train split passes `problem` so any custom row-level oversampling can
    # kick in via problem.apply_train_oversampling (default = identity).
    train_ds = make_tf_dataset(train, batch_size=args.batch_size, shuffle=True, problem=problem)
    val_ds   = make_tf_dataset(val,   batch_size=args.batch_size, shuffle=False)

    learner = OperatorLearner(
        problem=problem,
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr=args.lr,
        x_mean=train.get("X_mean"),
        x_std=train.get("X_std"),
    )
    if args.weights is not None:
        learner.load_weights(args.weights)

    dummy = tf.zeros((1, problem.input_dim), dtype=tf.float32)
    _ = learner.net(dummy, training=False)
    coef_dim = int(learner.basis.coef_dim)
    print("\n" + "=" * 60)
    print(f"OperatorNet: input_dim={problem.input_dim}, "
          f"state_dim={problem.state_dim}, N_p={coef_dim}, basis={learner.basis_name}")
    print("=" * 60)
    learner.net.summary(line_length=90)
    print("=" * 60 + "\n")

    use_schedule = USE_LR_SCHEDULE and (not args.no_lr_schedule)

    snapshot = {
        "problem": problem.name,
        "basis": learner.basis_name,
        "seed": int(args.seed),
        "timestamp": timestamp,
        "input_dim": int(INPUT_DIM),
        "hidden": int(args.hidden),
        "n_layers": int(args.n_layers),
        "T_final": float(DT_SEG),
        "Nt": int(NT_SEG),
        "N_p": coef_dim,
        "max_order": int(MAX_ORDER),
        "gamma_lpa": float(GAMMA),
        "adaf_N_m": int(ADAF_N_M),
        "adaf_N_p": int(ADAF_N_P),
        "physics_epochs": int(args.epochs),
        "physics_lr": float(args.lr),
        "physics_lr_min": float(args.lr_min),
        "lr_schedule": "cosine_warmup" if use_schedule else "fixed",
        "warmup_frac": float(args.warmup_frac),
        "rollout_epochs": int(args.rollout_epochs),
        "rollout_steps": int(args.rollout_steps),
        "rollout_alpha": float(args.rollout_alpha),
        "rollout_lr": float(args.rollout_lr),
        "batch_size": int(args.batch_size),
        "N_train": int(train["X"].shape[0]),
        "N_val": int(val["X"].shape[0]),
        "x_mean": train["X_mean"].tolist() if "X_mean" in train else None,
        "x_std":  train["X_std"].tolist()  if "X_std"  in train else None,
        "cons_w": float(problem.cons_w),
    }
    save_json(snapshot, os.path.join(run_dir, "config_snapshot.json"))
    print(f"[SAVE] config snapshot -> {os.path.join(run_dir, 'config_snapshot.json')}")

    fit_info = learner.fit(
        train_ds, val_ds, epochs=args.epochs, checkpoint_dir=ckpt_dir,
        use_lr_schedule=use_schedule,
        warmup_fraction=args.warmup_frac,
        min_lr=args.lr_min,
    )

    rollout_info = {}
    if args.rollout_epochs > 0:
        print(f"\n[rollout-DIAGNOSTIC] epochs={args.rollout_epochs}  K={args.rollout_steps}  "
              f"alpha={args.rollout_alpha}  lr={args.rollout_lr:.2e}")
        rollout_info = learner.fit_rollout(
            train_ds, val_ds,
            epochs=args.rollout_epochs,
            K=args.rollout_steps,
            alpha=args.rollout_alpha,
            fixed_lr=args.rollout_lr,
            checkpoint_dir=ckpt_dir,
        )

    history_npz = os.path.join(run_dir, "history.npz")
    np.savez_compressed(history_npz,
                        **{k: np.asarray(v) for k, v in learner.history.items() if len(v) > 0})
    print(f"[SAVE] history -> {history_npz}")
    plot_training_curves(learner.history, save_path=os.path.join(run_dir, "history.png"))

    summary = {
        "problem": problem.name,
        "run_dir": run_dir,
        "lr_schedule": "cosine_warmup" if use_schedule else "fixed",
        "peak_lr": args.lr, "min_lr": args.lr_min,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "weights_init": args.weights,
        **fit_info, **rollout_info,
    }
    save_json(summary, os.path.join(run_dir, "train_summary.json"))
    print(f"[SAVE] summary -> {os.path.join(run_dir, 'train_summary.json')}")
    if fit_info.get("best_checkpoint"):
        print(f"[BEST] checkpoint -> {fit_info['best_checkpoint']}")


if __name__ == "__main__":
    main()
