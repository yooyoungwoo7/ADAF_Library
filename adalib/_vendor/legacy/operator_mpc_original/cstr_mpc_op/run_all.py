#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
============================================================
End-to-end batch driver: trains every registered problem (or a chosen subset)
and runs eval immediately after, producing the same plots `main_eval.py` would.
Writes a summary table at the end.

Sequential mode (default, --parallel 1) hot-swaps the PROBLEM env var between
problems within the same process, re-importing config each time.

Parallel mode (--parallel N ≥ 2) launches one subprocess per problem
(`python main_train.py --problem <name>`) and runs up to N concurrently.
Each subprocess has its own TF graph + config, so there is no
inter-problem interference. Per-problem stdout/stderr go to a log file.

Usage:
    USE_GPU=0 python run_all.py                              # all 4, sequential
    USE_GPU=0 python run_all.py --parallel 4                 # 4 in parallel
    USE_GPU=0 python run_all.py --problems lotka cstr        # subset
    USE_GPU=0 python run_all.py --rebuild_data               # force fresh npz
    USE_GPU=0 python run_all.py --epochs 200                 # quick smoke
    USE_GPU=0 python run_all.py --skip_train                 # eval-only on
                                                              # latest run/problem
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import tensorflow as tf

if os.environ.get("USE_GPU", "0") != "1":
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


# ============================================================
# Helpers
# ============================================================
def _reset_modules():
    """Drop config/problems/data/models/utils so the next import sees the
    new PROBLEM env."""
    for mod in list(sys.modules):
        if mod == "config" or mod.startswith(("problems", "data", "models", "utils")):
            del sys.modules[mod]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(msg: str):
    print(f"\n[run_all] {msg}\n" + "─" * 70)


# ============================================================
# Per-problem train + eval
# ============================================================
def train_one(problem_name: str, args, seed: int) -> dict:
    """Run physics-only training for one problem. Returns the train_summary dict."""
    _log(f"TRAIN  problem={problem_name}  seed={seed}")
    os.environ["PROBLEM"] = problem_name
    _reset_modules()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import config as _cfg  # noqa: F401
    from problems.registry import get_problem
    from data.dataset_builder import (
        build_and_save_fullcase, build_segments_from_fullcase,
        compute_input_stats, save_segments,
        load_segments, make_tf_dataset,
    )
    from models.learner import OperatorLearner
    from utils.io_utils import ensure_dir, save_json
    from utils.plotting import plot_training_curves

    problem = get_problem(problem_name)
    paths = _cfg.get_data_paths()

    if args.rebuild_data:
        for k in paths:
            if os.path.exists(paths[k]):
                os.remove(paths[k])

    # Build fullcase + segment npz if missing
    if not all(os.path.exists(paths[k]) for k in ("train_fullcase", "val_fullcase", "test_fullcase")):
        build_and_save_fullcase(problem, paths["train_fullcase"], _cfg.N_TRAIN_CASES, seed=seed)
        build_and_save_fullcase(problem, paths["val_fullcase"],   _cfg.N_VAL_CASES,   seed=seed + 1)
        build_and_save_fullcase(problem, paths["test_fullcase"],  _cfg.N_TEST_CASES,  seed=seed + 2)

    if not all(os.path.exists(paths[k]) for k in ("train_segment", "val_segment", "test_segment")):
        train_seg = build_segments_from_fullcase(problem, paths["train_fullcase"], seed=seed + 100)
        X_mean, X_std = compute_input_stats(train_seg["X"])
        save_segments(paths["train_segment"], train_seg, X_mean=X_mean, X_std=X_std)
        val_seg = build_segments_from_fullcase(problem, paths["val_fullcase"], seed=seed + 101)
        save_segments(paths["val_segment"], val_seg, X_mean=X_mean, X_std=X_std)
        test_seg = build_segments_from_fullcase(problem, paths["test_fullcase"], seed=seed + 102)
        save_segments(paths["test_segment"], test_seg, X_mean=X_mean, X_std=X_std)

    timestamp = _stamp()
    run_dir = os.path.join(_cfg.get_result_dir(), f"{problem.name}_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    train = load_segments(paths["train_segment"])
    val   = load_segments(paths["val_segment"])
    train_ds = make_tf_dataset(train, batch_size=args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(val,   batch_size=args.batch_size, shuffle=False)

    learner = OperatorLearner(
        problem=problem, hidden=_cfg.HIDDEN, n_layers=_cfg.N_LAYERS,
        lr=args.lr, x_mean=train.get("X_mean"), x_std=train.get("X_std"),
        basis_name=getattr(args, "basis", None),
    )
    dummy = tf.zeros((1, problem.input_dim), dtype=tf.float32)
    _ = learner.net(dummy, training=False)

    snapshot = {
        "problem": problem.name,
        "basis": learner.basis_name,
        "seed": int(seed),
        "timestamp": timestamp,
        "input_dim": int(_cfg.INPUT_DIM),
        "hidden": int(_cfg.HIDDEN),
        "n_layers": int(_cfg.N_LAYERS),
        "T_final": float(_cfg.DT_SEG),
        "Nt": int(_cfg.NT_SEG),
        "N_p": int(learner.basis.coef_dim),
        "max_order": int(_cfg.MAX_ORDER),
        "gamma_lpa": float(_cfg.GAMMA),
        "adaf_N_m": int(_cfg.ADAF_N_M),
        "adaf_N_p": int(_cfg.ADAF_N_P),
        "physics_epochs": int(args.epochs),
        "physics_lr": float(args.lr),
        "physics_lr_min": float(_cfg.LR_MIN),
        "lr_schedule": "cosine_warmup" if _cfg.USE_LR_SCHEDULE else "fixed",
        "warmup_frac": float(_cfg.LR_WARMUP_FRACTION),
        "batch_size": int(args.batch_size),
        "N_train": int(train["X"].shape[0]),
        "N_val":   int(val["X"].shape[0]),
        "x_mean": train["X_mean"].tolist() if "X_mean" in train else None,
        "x_std":  train["X_std"].tolist()  if "X_std"  in train else None,
        "cons_w": float(problem.cons_w),
    }
    save_json(snapshot, os.path.join(run_dir, "config_snapshot.json"))

    fit_info = learner.fit(
        train_ds, val_ds, epochs=args.epochs, checkpoint_dir=ckpt_dir,
        use_lr_schedule=_cfg.USE_LR_SCHEDULE,
        warmup_fraction=_cfg.LR_WARMUP_FRACTION,
        min_lr=_cfg.LR_MIN,
    )

    history_npz = os.path.join(run_dir, "history.npz")
    np.savez_compressed(history_npz,
                        **{k: np.asarray(v) for k, v in learner.history.items() if len(v) > 0})
    plot_training_curves(learner.history, save_path=os.path.join(run_dir, "history.png"))

    summary = {
        "problem": problem.name, "run_dir": run_dir,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "peak_lr": args.lr, "min_lr": _cfg.LR_MIN,
        **fit_info,
    }
    save_json(summary, os.path.join(run_dir, "train_summary.json"))
    return summary


# ============================================================
# Subprocess-based training (parallel mode)
# ============================================================
def _threads_per_subprocess(parallel: int) -> int:
    """Given the parallelism level, decide how many CPU threads each child
    process should be allowed to use. Avoids oversubscription where N parallel
    TF processes each spawn `n_cores` threads → N·n_cores threads on n_cores."""
    try:
        n_cores = os.cpu_count() or 1
    except Exception:
        n_cores = 1
    return max(1, n_cores // max(1, parallel))


def train_one_subprocess(problem_name: str, args, log_dir: str) -> dict:
    """Spawn `main_train.py --problem <name>` as a child process.

    Designed for `--parallel N ≥ 2`: each problem runs in its own process so
    there is no interference between TF graphs / configs. stdout/stderr go to
    a log file under `log_dir`.

    To prevent thread oversubscription, the child's TF/BLAS thread pools are
    capped at `cpu_count // parallel` via env vars.
    """
    cmd = [sys.executable, "main_train.py", "--problem", problem_name]
    if args.epochs is not None:     cmd += ["--epochs",     str(args.epochs)]
    if args.batch_size is not None: cmd += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:         cmd += ["--lr",         str(args.lr)]
    if args.seed is not None:       cmd += ["--seed",       str(int(args.seed))]
    if args.rebuild_data:           cmd += ["--rebuild_data"]
    if getattr(args, "basis", None): cmd += ["--basis", args.basis]

    n_threads = _threads_per_subprocess(args.parallel)
    log_path = os.path.join(log_dir, f"train_{problem_name}.log")
    env = {
        **os.environ,
        "PROBLEM": problem_name,
        # cap each child's thread pools so that N parallel processes don't
        # collectively spawn N·n_cores BLAS / TF threads.
        "TF_NUM_INTEROP_THREADS": str(n_threads),
        "TF_NUM_INTRAOP_THREADS": str(n_threads),
        "OMP_NUM_THREADS":        str(n_threads),
        "MKL_NUM_THREADS":        str(n_threads),
        "OPENBLAS_NUM_THREADS":   str(n_threads),
        "VECLIB_MAXIMUM_THREADS": str(n_threads),
    }
    print(f"[run_all] start {problem_name:14s}  threads={n_threads}  "
          f"log → {os.path.basename(log_path)}")
    t0 = time.perf_counter()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0
    status = "OK " if result.returncode == 0 else "ERR"
    print(f"[run_all] {status} {problem_name:14s}  {elapsed:6.1f}s  rc={result.returncode}")
    return {
        "problem": problem_name,
        "returncode": int(result.returncode),
        "log_path": log_path,
        "train_seconds": elapsed,
        "run_dir": _latest_run_dir(problem_name),
        "skipped": False,
    }


def eval_one(problem_name: str, run_dir: str, weight_path: str, args) -> dict:
    """Run the full validation report (per-case rollout, comparison, sweeps,
    random cases). Returns a small summary dict."""
    _log(f"EVAL   problem={problem_name}  ckpt={os.path.basename(weight_path)}")
    os.environ["PROBLEM"] = problem_name
    _reset_modules()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import config as _cfg
    from problems.registry import get_problem
    from data.dataset_builder import load_segments, load_fullcase
    from models.learner import OperatorLearner
    from utils.io_utils import ensure_dir, save_json, save_csv_rows
    from utils.metrics import l2_rel, statewise_l2_rel
    from utils.plotting import plot_state_vs_reference, plot_residual_profile
    from utils.sweep_plots import plot_param_sweeps, plot_val_comparison, plot_random_cases

    problem = get_problem(problem_name)
    train = load_segments(_cfg.get_data_paths()["train_segment"])

    # Load basis used during training from the snapshot, so eval reconstructs
    # an identical learner regardless of the current BASIS env.
    snap_basis = None
    snap_path = os.path.join(run_dir, "config_snapshot.json")
    if os.path.exists(snap_path):
        try:
            with open(snap_path) as f:
                snap_basis = json.load(f).get("basis")
        except Exception:
            snap_basis = None

    learner = OperatorLearner(
        problem=problem, hidden=_cfg.HIDDEN, n_layers=_cfg.N_LAYERS,
        x_mean=train.get("X_mean"), x_std=train.get("X_std"),
        basis_name=snap_basis,
    )
    learner.load_weights(weight_path)

    out_dir = os.path.join(run_dir, "eval_phys_best")
    ensure_dir(out_dir)

    full = load_fullcase(_cfg.get_data_paths()["val_fullcase"])
    X0 = np.asarray(full["X0"], dtype=np.float32)
    n_cases = min(int(args.n_cases), X0.shape[0])

    rows = [["case_idx", "l2_rel_total"] + [f"l2_{lab}" for lab in problem.state_labels]]
    per_case = []
    for i in range(n_cases):
        tag = f"case_{i:03d}"
        rollout = learner.rollout_full_trajectory(X0[i], n_seg=_cfg.N_SEG)
        t = rollout["t"]; pred = rollout["x"]; residual = rollout["residual"]
        x0  = X0[i, :problem.state_dim]
        th  = X0[i, problem.state_dim:problem.state_dim + problem.param_dim]
        ref = problem.solve_reference(theta=th, x0=x0, t_grid=t)

        tfac = getattr(problem, "time_factor", 1.0)
        plot_state_vs_reference(
            t, pred, ref, labels=problem.state_plot_labels(),
            title_prefix=tag, subtitle=problem.case_subtitle(X0[i]),
            save_path=os.path.join(out_dir, f"{tag}_traj.png"),
            time_unit=problem.time_unit,
            extras=problem.extra_plot_traces() or None,
            theta=th,
            time_factor=tfac,
        )
        plot_residual_profile(
            t, residual, labels=tuple(f"res_{lab}" for lab in problem.state_labels),
            save_path=os.path.join(out_dir, f"{tag}_residual.png"),
            time_unit=problem.time_unit,
            time_factor=tfac,
        )
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}_rollout.npz"),
            x_input=X0[i], t=t, x_pred=pred, x_ref=ref, residual=residual,
        )
        l2_total = l2_rel(pred, ref)
        l2_states = statewise_l2_rel(pred, ref).tolist()
        rows.append([i, l2_total] + l2_states)
        per_case.append({"l2_rel_total": l2_total, "l2_rel_states": l2_states})
        print(f"  [VAL] {tag}  L2_rel={l2_total:.3e}  per-state={[f'{v:.2e}' for v in l2_states]}")

    save_csv_rows(rows[1:], os.path.join(out_dir, "validation_metrics.csv"), header=rows[0])
    save_json({"per_case": per_case}, os.path.join(out_dir, "validation_summary.json"))

    if args.comparison_n > 0:
        plot_val_comparison(learner, _cfg.get_data_paths()["val_fullcase"],
                            os.path.join(out_dir, "comparison.png"),
                            n_cases=args.comparison_n, seed=args.random_seed or 0)
    if args.sweep_n > 0:
        plot_param_sweeps(learner, os.path.join(out_dir, "param_sweeps.png"),
                          n=args.sweep_n)
    if args.random_n > 0:
        plot_random_cases(learner, os.path.join(out_dir, "random_cases.png"),
                          n_cases=args.random_n, seed=args.random_seed)

    avg_l2 = float(np.mean([c["l2_rel_total"] for c in per_case])) if per_case else float("nan")
    return {
        "problem": problem.name,
        "out_dir": out_dir,
        "n_cases_eval": n_cases,
        "avg_l2_rel": avg_l2,
        "per_case": per_case,
    }


# ============================================================
# Driver
# ============================================================
def _latest_run_dir(problem_name: str) -> str | None:
    """Pick the most recent results/<problem>/<problem>_<ts>/ directory."""
    base = os.path.join("results", problem_name)
    if not os.path.isdir(base):
        return None
    runs = sorted(d for d in os.listdir(base)
                  if d.startswith(f"{problem_name}_") and os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, runs[-1]) if runs else None


def _best_phys_ckpt(run_dir: str) -> str | None:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    bests = sorted(f for f in os.listdir(ckpt_dir)
                   if f.endswith("_best.weights.h5"))
    return os.path.join(ckpt_dir, bests[-1]) if bests else None


def main():
    parser = argparse.ArgumentParser(
        description="Sequential train+eval for all (or chosen) registered problems."
    )
    parser.add_argument("--problems", nargs="*", default=None,
                        help="Subset of problem names. Default: all 4.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override PHYSICS_EPOCHS for every problem (default: per-problem config).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override PHYSICS_BATCH_SIZE.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override PHYSICS_LR.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rebuild_data", action="store_true")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training; eval the latest existing run for each problem.")
    parser.add_argument("--n_cases", type=int, default=5,
                        help="# val cases to roll out per problem.")
    parser.add_argument("--comparison_n", type=int, default=4)
    parser.add_argument("--sweep_n", type=int, default=8)
    parser.add_argument("--random_n", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of problems to train concurrently in subprocesses "
                             "(1 = sequential in-process, default).")
    parser.add_argument("--basis", default=None,
                        help="Basis to use for every problem: 'lpa' or 'adaf' (default: BASIS env / 'lpa').")
    args = parser.parse_args()

    # Determine the list of problems
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from problems.registry import list_problems
    all_problems = list_problems()
    todo = args.problems or all_problems
    bad = [p for p in todo if p not in all_problems]
    if bad:
        raise ValueError(f"Unknown problems {bad}. Registered: {all_problems}")

    if args.seed is None:
        args.seed = secrets.randbits(31)
    print(f"[run_all] seed={args.seed}  problems={todo}")

    summaries = []
    overall_t0 = time.perf_counter()

    # ── Training phase ────────────────────────────────────────
    train_summaries: dict[str, dict] = {}
    if args.skip_train:
        # Eval-only on the latest existing run dir for each problem
        for pname in todo:
            run_dir = _latest_run_dir(pname)
            if run_dir is None:
                print(f"[run_all] no existing run for {pname}, skipping eval too.")
                continue
            train_summaries[pname] = {"problem": pname, "run_dir": run_dir, "skipped": True}
    elif args.parallel >= 2:
        # Parallel mode: each problem in its own subprocess, ThreadPoolExecutor
        # waits on up to `args.parallel` at a time.
        # Leading underscore so the parallel-log dir sorts/visually separates
        # from the per-problem `<problem>_<timestamp>` run dirs.
        log_dir = os.path.join("results", f"_run_all_parallel_{_stamp()}")
        os.makedirs(log_dir, exist_ok=True)
        _log(f"PARALLEL TRAIN  workers={args.parallel}  problems={todo}  log_dir={log_dir}")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(train_one_subprocess, p, args, log_dir): p
                for p in todo
            }
            for fut in as_completed(futures):
                summary = fut.result()
                if summary["returncode"] != 0:
                    print(f"[run_all] WARNING: {summary['problem']} subprocess failed (rc={summary['returncode']}); "
                          f"see {summary['log_path']}")
                train_summaries[summary["problem"]] = summary
    else:
        # Sequential in-process mode (default). Same process, hot-swap PROBLEM env.
        for pname in todo:
            os.environ["PROBLEM"] = pname
            if args.basis is not None:
                os.environ["BASIS"] = args.basis.lower()
            _reset_modules()
            import config as _cfg
            epochs = args.epochs if args.epochs is not None else _cfg.PHYSICS_EPOCHS
            batch  = args.batch_size if args.batch_size is not None else _cfg.PHYSICS_BATCH_SIZE
            lr     = args.lr if args.lr is not None else _cfg.PHYSICS_LR

            np.random.seed(args.seed)
            tf.random.set_seed(args.seed)

            class _A: pass
            _a = _A()
            _a.epochs = epochs; _a.batch_size = batch; _a.lr = lr
            _a.rebuild_data = args.rebuild_data
            _a.basis = args.basis
            t0 = time.perf_counter()
            ts = train_one(pname, _a, seed=args.seed)
            ts["train_seconds"] = time.perf_counter() - t0
            train_summaries[pname] = ts

    # ── Eval phase (always sequential, in-process) ────────────
    for pname in todo:
        train_summary = train_summaries.get(pname)
        if train_summary is None:
            continue
        run_dir = train_summary["run_dir"]
        if run_dir is None:
            print(f"[run_all] {pname}: no run_dir, skipping eval.")
            summaries.append(dict(train=train_summary, eval=None))
            continue
        ckpt = _best_phys_ckpt(run_dir)
        if ckpt is None:
            print(f"[run_all] {pname}: no best ckpt found in {run_dir}, skipping eval.")
            summaries.append(dict(train=train_summary, eval=None))
            continue
        eval_summary = eval_one(pname, run_dir, ckpt, args)
        summaries.append(dict(train=train_summary, eval=eval_summary))

    overall_seconds = time.perf_counter() - overall_t0

    # ── Final report ──────────────────────────────────────
    _log(f"DONE  total wall time = {overall_seconds:.1f}s")
    print(f"{'problem':14s}  {'avg L2_rel':>11s}  {'best val_phys':>14s}  {'train s':>9s}  run_dir")
    for s in summaries:
        t = s["train"]; e = s["eval"]
        # If subprocess mode, train_summary fields aren't populated in `t` —
        # read the canonical train_summary.json the child wrote to run_dir.
        bvp_val = t.get("best_val_phys_loss")
        if bvp_val is None and t.get("run_dir"):
            ts_path = os.path.join(t["run_dir"], "train_summary.json")
            if os.path.exists(ts_path):
                try:
                    with open(ts_path) as f:
                        bvp_val = json.load(f).get("best_val_phys_loss")
                except Exception:
                    pass
        avg = f"{e['avg_l2_rel']:.3e}" if e else "—"
        bvp = f"{bvp_val:.3e}" if bvp_val is not None else "—"
        ts  = f"{t.get('train_seconds', 0.0):.0f}" if not t.get("skipped") else "skipped"
        print(f"{t['problem']:14s}  {avg:>11s}  {bvp:>14s}  {ts:>9s}  {t['run_dir']}")

    # Persist a top-level batch summary
    batch_summary = {
        "timestamp": _stamp(),
        "seed": int(args.seed),
        "total_seconds": overall_seconds,
        "problems": [s["train"]["problem"] for s in summaries],
        "results": summaries,
    }
    out_path = os.path.join("results", f"run_all_{_stamp()}.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(batch_summary, f, indent=2, default=str)
    print(f"\n[run_all] batch summary -> {out_path}")


if __name__ == "__main__":
    main()
