"""
Surrogate-MPC showcase: differentiable + batched operator inference.

Runs the same trained operator surrogate through three MPC optimizers and
reports a head-to-head comparison supporting two claims:

  1. Differentiable inference — SLSQP with the exact dJ/du obtained by
     automatic differentiation through the operator ("autodiff") versus
     the same rollout cost with finite-difference gradients ("fd").
     FD needs (1 + H*n_u) rollouts per gradient; autodiff needs one
     forward + one backward pass regardless of H*n_u.

  2. Batched inference — CEM sampling MPC evaluates B candidate control
     sequences per batched forward pass. A numerical integrator must
     integrate candidates one by one. A throughput microbenchmark
     (operator batch vs solve_ivp loop) quantifies the gap.

Run from adalib_project/ root:
    python examples/mpc/surrogate_mpc_showcase.py --system triple_tank \
        --horizon 5 --n_steps 15
    python examples/mpc/surrogate_mpc_showcase.py --system cstr \
        --horizon 5 --n_steps 15 --reuse     # after the first run
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

import adalib
from adalib.mpc.options import MPCOptions

SYSTEMS = {
    "cstr": {
        "x0":       [0.8, 0.5, 141.0, 141.0],
        "target":   {"T_R": 136.0},
        "mpc_name": "cstr_mpc",
        "n_u":      2 - 1,  # 1 control (Q)
    },
    "triple_tank": {
        "x0":       [190.0, 100.0, 140.0],   # regulation: interior optimum
        "target":   {"h3": 150.0},
        "mpc_name": "triple_tank_mpc",
        "n_u":      2,      # Q1, Q2
    },
}


def make_opts(args, work_dir, first_run, **overrides):
    opts = MPCOptions(
        mode="tracking",
        basis="lpa",
        target=dict(SYSTEMS[args.system]["target"]),
        n_steps=args.n_steps,
        horizon=args.horizon,
        n_train=args.n_train,
        n_val=max(args.n_train // 5, 8),
        epochs=args.epochs,
        hidden=128,
        n_layers=3,
        work_dir=work_dir,
        verbose=args.verbose,
        generate_data=first_run,
        reuse_existing_data=not first_run,
        train_operator=first_run,
        reuse_existing_operator=not first_run,
    )
    for k, v in overrides.items():
        setattr(opts, k, v)
    return opts


def tracking_error(result, target, state_labels):
    """Mean |y - y_ref| over the second half of the closed loop."""
    errs = []
    for name, ref in target.items():
        j = list(state_labels).index(name)
        y = result.x[len(result.x) // 2:, j]
        errs.append(float(np.mean(np.abs(y - ref))))
    return float(np.mean(errs))


def rebuild_learner(args, work_dir, ckpt):
    """Reload the trained operator for the throughput microbenchmark."""
    from adalib.workflows.mpc_workflow import _ensure_legacy_on_path
    from adalib.utils.legacy_context import reload_legacy_chain

    mpc_name = SYSTEMS[args.system]["mpc_name"]
    _ensure_legacy_on_path()
    reload_legacy_chain(mpc_name, "lpa")
    import problems.registry as _preg
    import models.learner as _ml
    import data.dataset_builder as _db

    problem = _preg.get_problem(mpc_name)
    seg = _db.load_segments(os.path.join(
        work_dir, "operator", "data", f"{mpc_name}_train_segments.npz"))
    learner = _ml.OperatorLearner(
        problem=problem, hidden=128, n_layers=3,
        x_mean=seg.get("X_mean"), x_std=seg.get("X_std"),
    )
    learner.load_weights(ckpt)
    return learner, problem


def throughput_benchmark(args, learner, problem, horizon):
    """Batched operator rollout vs sequential solve_ivp, same workload."""
    import tensorflow as tf
    from scipy.integrate import solve_ivp
    from adalib.mpc._surrogate_mpc import _build_tracking_spec, _make_rollout_fns

    opts = MPCOptions(target=dict(SYSTEMS[args.system]["target"]))
    mpc_name = SYSTEMS[args.system]["mpc_name"]
    spec = _build_tracking_spec(mpc_name, problem, opts)
    batch_cost, _ = _make_rollout_fns(learner, horizon, spec)

    import config as _cfg
    t_seg = float(_cfg.DT_SEG)
    DTYPE = learner.basis.dtype
    xk = tf.constant(np.asarray(SYSTEMS[args.system]["x0"], np.float32),
                     dtype=DTYPE)
    n_u = len(spec["u_lo"])
    rng = np.random.default_rng(0)

    print(f"\n── Batched-inference throughput  (horizon H={horizon}, "
          f"one candidate = {horizon} segments) " + "─" * 16)
    print(f"{'B':>6}  {'operator batch [ms]':>20}  {'per-candidate [ms]':>19}")
    rows = []
    for B in (1, 64, 512, 2048):
        Z = rng.uniform(0.0, 1.0, (B, horizon, n_u)).astype(np.float32)
        Zt = tf.constant(Z, dtype=DTYPE)
        batch_cost(xk, Zt)                       # compile / warm-up
        t0 = time.perf_counter()
        n_rep = 5
        for _ in range(n_rep):
            batch_cost(xk, Zt).numpy()
        ms = (time.perf_counter() - t0) / n_rep * 1000.0
        rows.append((B, ms))
        print(f"{B:>6}  {ms:>20.2f}  {ms / B:>19.4f}")

    # scipy: one candidate = H sequential segment integrations
    u_lo, u_hi = spec["u_lo"], spec["u_hi"]
    x0 = np.asarray(SYSTEMS[args.system]["x0"], np.float64)
    n_probe = 16
    t0 = time.perf_counter()
    for _ in range(n_probe):
        xc = x0.copy()
        for k in range(horizon):
            u = u_lo + rng.uniform(0, 1, n_u) * (u_hi - u_lo)
            if spec["plant_kind"] == "cstr":
                f = lambda t, y: problem._rhs(t, y, float(u[0]))
                method = "BDF"
            else:
                f = lambda t, y: problem._rhs(t, y, float(u[0]), float(u[1]))
                method = "RK45"
            sol = solve_ivp(f, (0.0, t_seg), xc, method=method,
                            rtol=1e-6, atol=1e-8)
            xc = sol.y[:, -1]
    scipy_ms = (time.perf_counter() - t0) / n_probe * 1000.0
    print(f"\n  solve_ivp per candidate: {scipy_ms:.2f} ms "
          f"({SYSTEMS[args.system]['mpc_name']}, {horizon} segments)")
    B_big, ms_big = rows[-1]
    speedup = scipy_ms / (ms_big / B_big)
    print(f"  → batched operator speedup at B={B_big}: {speedup:,.0f}× "
          f"per candidate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS), default="triple_tank")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing data + operator checkpoint")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = SYSTEMS[args.system]
    work_dir = os.path.abspath(f"./runs/{args.system}_surrogate_mpc")
    first_run = not args.reuse

    runs = []

    # 1) train operator (first run) + FD baseline
    print(f"\n=== [1/3] SLSQP finite-difference  (H={args.horizon}) ===")
    r_fd = adalib.run_mpc(system=args.system, x0=cfg["x0"],
                          options=make_opts(args, work_dir, first_run,
                                            gradient="fd"))
    runs.append(("SLSQP-fd", r_fd))
    first_run = False   # everything below reuses data + checkpoint

    # 2) autodiff gradients
    print(f"\n=== [2/3] SLSQP autodiff  (H={args.horizon}) ===")
    r_ad = adalib.run_mpc(system=args.system, x0=cfg["x0"],
                          options=make_opts(args, work_dir, False,
                                            gradient="autodiff"))
    runs.append(("SLSQP-autodiff", r_ad))

    # 3) CEM batched sampling
    print(f"\n=== [3/3] CEM batched sampling  (H={args.horizon}) ===")
    r_cem = adalib.run_mpc(system=args.system, x0=cfg["x0"],
                           options=make_opts(args, work_dir, False,
                                             optimizer="CEM"))
    runs.append(("CEM", r_cem))

    # ── summary table ─────────────────────────────────────────────────
    state_labels = {"cstr": ("C_A", "C_B", "T_R", "T_K"),
                    "triple_tank": ("h1", "h2", "h3")}[args.system]
    print(f"\n{'=' * 76}")
    print(f"Summary — {args.system}, horizon H={args.horizon}, "
          f"{args.n_steps} closed-loop steps")
    print(f"{'=' * 76}")
    hdr = (f"{'optimizer':<16} {'opt ms/step':>12} {'nfev/step':>10} "
           f"{'njev/step':>10} {'|y-ref| (2nd half)':>19}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in runs:
        m = r.metadata
        err = tracking_error(r, cfg["target"], state_labels)
        nfev = m.get("opt_nfev_mean", m.get("rollouts_per_step", float("nan")))
        njev = m.get("opt_njev_mean", float("nan"))
        print(f"{name:<16} {m.get('opt_ms_per_step_mean', float('nan')):>12.1f} "
              f"{nfev:>10.1f} {njev:>10.1f} {err:>19.4f}")

    # ── plots ─────────────────────────────────────────────────────────
    adalib.utils.set_adalib_plot_style()
    fig_path = os.path.join(work_dir, f"{args.system}_surrogate_mpc.png")
    adalib.utils.plot_mpc_result(
        [r for _, r in runs],
        labels=[n for n, _ in runs],
        target=cfg["target"],
        save_path=fig_path,
    )
    print(f"\nPlot saved → {fig_path}")

    # ── batched-inference microbenchmark ──────────────────────────────
    ckpt = r_ad.metadata.get("best_checkpoint") or \
        r_ad.operator_result.get("best_checkpoint")
    learner, problem = rebuild_learner(args, work_dir, ckpt)
    throughput_benchmark(args, learner, problem, args.horizon)


if __name__ == "__main__":
    main()
