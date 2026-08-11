"""
adalib/mpc/_surrogate_mpc.py

Tracking-MPC loops that exploit the two computational properties of the
trained operator surrogate (built-in ``cstr_mpc`` / ``triple_tank_mpc``
problems):

1. Differentiable inference  (``MPCOptions.gradient="autodiff"`` / ``"fd"``)
   The one-segment map (x_k, u_k) -> x_{k+1} is a pure TF graph
   (MLP -> W panel weights -> linear LPA/ADA-F basis), so the exact
   Jacobian of a horizon-H rollout cost w.r.t. the control sequence is
   available through automatic differentiation.  ``"autodiff"`` supplies
   this Jacobian to SLSQP; ``"fd"`` runs the *same* rollout cost with
   SLSQP finite differences for a controlled comparison.

2. Batched inference  (``MPCOptions.optimizer="CEM"``)
   One forward pass evaluates B candidate control sequences at once, so a
   sampling-based controller (cross-entropy method) evaluates B x H
   segment predictions with H batched network calls per iteration instead
   of B x H sequential ODE solves.

Controls are optimised in the normalised space z in [0,1]^{H*n_u} with
u = u_lo + z*(u_hi - u_lo) applied *inside* the TF graph, so both gradient
modes and CEM see a well-conditioned problem regardless of control units
(CSTR: Q in [-8500, 0] kJ/h).

The legacy 1-step loops in workflows/mpc_workflow.py are untouched and
remain the default (``gradient=None``, ``optimizer="SLSQP"``).
"""
from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════
# Tracking spec resolution (per built-in MPC problem)
# ══════════════════════════════════════════════════════════════════════

def _build_tracking_spec(mpc_name: str, problem, opts) -> Dict:
    """Resolve tracked states, reference, control/state boxes and weights."""
    state_labels = list(problem.state_labels)

    if mpc_name == "cstr_mpc":
        from problems.cstr_mpc_problem import Q_RANGE
        u_names = ["Q"]
        u_lo = np.array([Q_RANGE[0]], dtype=np.float64)
        u_hi = np.array([Q_RANGE[1]], dtype=np.float64)
        if opts.control_bounds is not None:
            cb = opts.control_bounds
            if isinstance(cb, (tuple, list)) and len(cb) == 2:
                u_lo[0], u_hi[0] = float(cb[0]), float(cb[1])
            elif isinstance(cb, dict):
                u_lo[0] = float(cb.get("Q_lo", u_lo[0]))
                u_hi[0] = float(cb.get("Q_hi", u_hi[0]))
        default_target = {"T_R": 136.0}
        x_min = np.array([0.0, 0.0,  50.0,  50.0], dtype=np.float64)
        x_max = np.array([5.0, 5.0, 200.0, 200.0], dtype=np.float64)
        plant_kind = "cstr"

    elif mpc_name == "triple_tank_mpc":
        from problems.triple_tank_mpc_problem import Q1_RANGE, Q2_RANGE
        u_names = ["Q1", "Q2"]
        u_lo = np.array([Q1_RANGE[0], Q2_RANGE[0]], dtype=np.float64)
        u_hi = np.array([Q1_RANGE[1], Q2_RANGE[1]], dtype=np.float64)
        if isinstance(opts.control_bounds, dict):
            cb = opts.control_bounds
            u_lo[0] = float(cb.get("Q1_lo", u_lo[0]))
            u_hi[0] = float(cb.get("Q1_hi", u_hi[0]))
            u_lo[1] = float(cb.get("Q2_lo", u_lo[1]))
            u_hi[1] = float(cb.get("Q2_hi", u_hi[1]))
        default_target = {"h3": 150.0}
        # The problem operates in cm (X0_UPPER = 300 cm); clip with headroom.
        x_min = np.zeros(3, dtype=np.float64)
        x_max = np.full(3, 350.0, dtype=np.float64)
        plant_kind = "triple_tank"

    else:
        raise NotImplementedError(
            f"Surrogate MPC (gradient/CEM) supports 'cstr_mpc' and "
            f"'triple_tank_mpc' only, got {mpc_name!r}."
        )

    target = opts.target if opts.target is not None else default_target

    # tracked state indices + reference vector
    if isinstance(target, (list, tuple)):
        tracked_idx = [i for i, v in enumerate(target) if v is not None]
        y_ref = [float(target[i]) for i in tracked_idx]
    else:
        if isinstance(target, dict):
            tgt = {k: float(v) for k, v in target.items()}
        else:  # scalar → first default key
            key = next(iter(default_target))
            tgt = {key: float(target)}
        if opts.controlled_variables:
            names = [n for n in opts.controlled_variables if n in state_labels]
        else:
            names = [n for n in state_labels if n in tgt]
        if not names:
            raise ValueError(
                f"target {target!r} matches no state label of {state_labels}."
            )
        tracked_idx = [state_labels.index(n) for n in names]
        y_ref = [tgt.get(n, float(default_target.get(n, 0.0))) for n in names]

    n_u = len(u_lo)
    q_w = np.asarray(opts.tracking_weights if opts.tracking_weights is not None
                     else np.ones(len(tracked_idx)), dtype=np.float64).ravel()
    r_w = np.asarray(opts.control_weights if opts.control_weights is not None
                     else np.zeros(n_u), dtype=np.float64).ravel()
    if len(q_w) != len(tracked_idx):
        q_w = np.ones(len(tracked_idx), dtype=np.float64)
    if len(r_w) != n_u:
        r_w = np.zeros(n_u, dtype=np.float64)

    return {
        "u_names":     u_names,
        "u_lo":        u_lo,
        "u_hi":        u_hi,
        "tracked_idx": tracked_idx,
        "y_ref":       np.asarray(y_ref, dtype=np.float64),
        "q_w":         q_w,
        "r_w":         r_w,
        "x_min":       x_min,
        "x_max":       x_max,
        "plant_kind":  plant_kind,
    }


# ══════════════════════════════════════════════════════════════════════
# Differentiable horizon rollout (TF graph through the operator)
# ══════════════════════════════════════════════════════════════════════

def _make_rollout_fns(learner, H: int, spec: Dict):
    """Build (batch_cost, cost_and_grad) tf.functions for a horizon-H rollout.

    batch_cost(xk, Z)      : xk (n_state,), Z (B, H, n_u) in [0,1] → J (B,)
    cost_and_grad(xk, z)   : z (H*n_u,) in [0,1] → (J scalar, dJ/dz (H*n_u,))
    """
    import tensorflow as tf

    DTYPE = learner.basis.dtype
    n_u   = len(spec["u_lo"])
    u_lo   = tf.constant(spec["u_lo"],                  dtype=DTYPE)
    u_span = tf.constant(spec["u_hi"] - spec["u_lo"],   dtype=DTYPE)
    idx    = tf.constant(spec["tracked_idx"],           dtype=tf.int32)
    y_ref  = tf.constant(spec["y_ref"],                 dtype=DTYPE)
    q_w    = tf.constant(spec["q_w"],                   dtype=DTYPE)
    r_w    = tf.constant(spec["r_w"],                   dtype=DTYPE)
    x_min  = tf.constant(spec["x_min"],                 dtype=DTYPE)
    x_max  = tf.constant(spec["x_max"],                 dtype=DTYPE)

    def _rollout_cost(x0_b, Z):
        """x0_b (B, n_state), Z (B, H, n_u) in [0,1] → J (B,)"""
        U  = u_lo[None, None, :] + Z * u_span[None, None, :]
        xc = x0_b
        J  = tf.zeros(tf.shape(Z)[0], dtype=DTYPE)
        for k in range(H):
            z_in = tf.concat([xc, U[:, k, :]], axis=1)
            out  = learner.forward_segment(z_in, training=False)
            xc   = tf.clip_by_value(out["x_end"], x_min, x_max)
            dy   = tf.gather(xc, idx, axis=1) - y_ref[None, :]
            J    = (J
                    + tf.reduce_sum(q_w * dy * dy, axis=1)
                    + tf.reduce_sum(r_w * tf.square(U[:, k, :]), axis=1))
        return J

    @tf.function(reduce_retracing=True)
    def batch_cost(xk, Z):
        x0_b = tf.repeat(xk[None, :], tf.shape(Z)[0], axis=0)
        return _rollout_cost(x0_b, Z)

    @tf.function(reduce_retracing=True)
    def cost_and_grad(xk, z_flat):
        with tf.GradientTape() as tape:
            tape.watch(z_flat)
            Z = tf.reshape(z_flat, (1, H, n_u))
            J = _rollout_cost(xk[None, :], Z)[0]
        g = tape.gradient(J, z_flat)
        return J, g

    return batch_cost, cost_and_grad


# ══════════════════════════════════════════════════════════════════════
# Plant step (true ODE via scipy) per built-in problem
# ══════════════════════════════════════════════════════════════════════

def _make_plant_step(problem, spec: Dict, t_seg: float):
    from scipy.integrate import solve_ivp

    x_min = spec["x_min"].astype(np.float32)
    x_max = spec["x_max"].astype(np.float32)

    if spec["plant_kind"] == "cstr":
        method = "BDF"          # stiff Arrhenius kinetics
        def ode(u_vec):
            return lambda t, y: problem._rhs(t, y, float(u_vec[0]))
    else:                        # triple_tank: _rhs(t, x, Q1, Q2)
        method = "RK45"
        def ode(u_vec):
            return lambda t, y: problem._rhs(t, y, float(u_vec[0]),
                                             float(u_vec[1]))

    def plant_step(xk, u_vec):
        sol = solve_ivp(
            ode(u_vec),
            t_span=(0.0, t_seg),
            y0=np.asarray(xk, dtype=np.float64),
            method=method,
            rtol=1e-6, atol=1e-8,
        )
        return np.clip(sol.y[:, -1].astype(np.float32), x_min, x_max)

    return plant_step


# ══════════════════════════════════════════════════════════════════════
# Closed-loop: SLSQP with autodiff or finite-difference gradients
# ══════════════════════════════════════════════════════════════════════

def _run_gradient_mpc(learner, problem, spec, x0, opts, t_seg):
    import tensorflow as tf
    from scipy.optimize import minimize

    DTYPE   = learner.basis.dtype
    H       = max(1, int(opts.horizon or 1))
    n_u     = len(spec["u_lo"])
    n_steps = int(opts.n_steps)
    use_ad  = (opts.gradient or "autodiff").lower() == "autodiff"

    batch_cost, cost_and_grad = _make_rollout_fns(learner, H, spec)
    plant_step = _make_plant_step(problem, spec, t_seg)

    u_lo, u_hi = spec["u_lo"], spec["u_hi"]
    u_span     = u_hi - u_lo
    bounds01   = [(0.0, 1.0)] * (H * n_u)
    slsqp_opts = {"ftol": 1e-8, "maxiter": 200}
    if not use_ad:
        # Sized for a float32 surrogate: SLSQP's default absolute step
        # (~1.5e-8) is below the model's numerical noise floor.
        slsqp_opts["eps"] = 1e-3

    xk    = np.asarray(x0, dtype=np.float32)
    z0    = np.full(H * n_u, 0.5, dtype=np.float64)
    x_log = [xk.copy()]
    u_log: List[np.ndarray] = []
    J_log: List[float] = []
    nit_l, nfev_l, njev_l, ms_l = [], [], [], []

    for step in range(n_steps):
        xk_tf = tf.constant(xk, dtype=DTYPE)
        t0 = time.perf_counter()

        if use_ad:
            def fun(z):
                J, g = cost_and_grad(xk_tf, tf.constant(z, dtype=DTYPE))
                return float(J.numpy()), np.asarray(g.numpy(), dtype=np.float64)
            res = minimize(fun, z0, jac=True, method="SLSQP",
                           bounds=bounds01, options=slsqp_opts)
        else:
            def fun(z):
                Z = tf.constant(z.reshape(1, H, n_u), dtype=DTYPE)
                return float(batch_cost(xk_tf, Z).numpy()[0])
            res = minimize(fun, z0, method="SLSQP",
                           bounds=bounds01, options=slsqp_opts)

        ms_l.append((time.perf_counter() - t0) * 1000.0)
        nit_l.append(int(res.nit))
        nfev_l.append(int(res.nfev))
        njev_l.append(int(getattr(res, "njev", 0) or 0))

        z_opt = np.clip(res.x, 0.0, 1.0).reshape(H, n_u)
        u_seq = u_lo[None, :] + z_opt * u_span[None, :]
        u_apply = u_seq[0]

        xk = plant_step(xk, u_apply)
        x_log.append(xk.copy())
        u_log.append(u_apply.astype(np.float32))
        J_log.append(float(res.fun))

        # warm start: shift the optimal sequence one segment forward
        z0 = np.concatenate([z_opt[1:], z_opt[-1:]], axis=0).reshape(-1)

        if opts.verbose:
            u_str = ", ".join(f"{n}={v:.1f}"
                              for n, v in zip(spec["u_names"], u_apply))
            y_now = xk[spec["tracked_idx"]]
            print(f"  [step {step+1:3d}]  {u_str}  "
                  f"y={np.round(y_now, 3).tolist()}  J={res.fun:.3e}  "
                  f"nit={res.nit}  nfev={res.nfev}  "
                  f"{ms_l[-1]:.0f} ms")

    def _mean_tail(v):  # exclude step 0 (tf.function compile) when possible
        return float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))

    stats = {
        "mpc_optimizer":        "SLSQP-autodiff" if use_ad else "SLSQP-fd",
        "mpc_horizon":          H,
        "opt_nit_mean":         float(np.mean(nit_l)),
        "opt_nfev_mean":        float(np.mean(nfev_l)),
        "opt_njev_mean":        float(np.mean(njev_l)),
        "opt_ms_first_step":    float(ms_l[0]),
        "opt_ms_per_step_mean": _mean_tail(ms_l),
    }
    return x_log, u_log, J_log, stats


# ══════════════════════════════════════════════════════════════════════
# Closed-loop: cross-entropy method (batched surrogate rollouts)
# ══════════════════════════════════════════════════════════════════════

def _run_cem_mpc(learner, problem, spec, x0, opts, t_seg):
    import tensorflow as tf

    DTYPE   = learner.basis.dtype
    H       = max(1, int(opts.horizon or 1))
    n_u     = len(spec["u_lo"])
    n_steps = int(opts.n_steps)
    B       = int(opts.cem_samples)
    n_el    = min(int(opts.cem_elites), B)
    iters   = int(opts.cem_iters)
    alpha   = float(opts.cem_alpha)

    batch_cost, _ = _make_rollout_fns(learner, H, spec)
    plant_step = _make_plant_step(problem, spec, t_seg)

    u_lo, u_hi = spec["u_lo"], spec["u_hi"]
    u_span     = u_hi - u_lo
    rng        = np.random.default_rng(opts.seed)

    xk    = np.asarray(x0, dtype=np.float32)
    mean  = np.full((H, n_u), 0.5, dtype=np.float64)
    std0  = 0.25
    x_log = [xk.copy()]
    u_log: List[np.ndarray] = []
    J_log: List[float] = []
    ms_l  = []

    for step in range(n_steps):
        xk_tf = tf.constant(xk, dtype=DTYPE)
        std = np.full((H, n_u), std0, dtype=np.float64)
        t0 = time.perf_counter()

        best_J = np.inf
        for _ in range(iters):
            Z = np.clip(mean[None] + std[None] * rng.standard_normal((B, H, n_u)),
                        0.0, 1.0)
            J = batch_cost(xk_tf, tf.constant(Z, dtype=DTYPE)).numpy()
            order = np.argsort(J)
            elites = Z[order[:n_el]]
            best_J = min(best_J, float(J[order[0]]))
            mean = alpha * elites.mean(axis=0) + (1.0 - alpha) * mean
            std  = alpha * elites.std(axis=0)  + (1.0 - alpha) * std
            std  = np.maximum(std, 0.01)

        ms_l.append((time.perf_counter() - t0) * 1000.0)

        u_seq   = u_lo[None, :] + mean * u_span[None, :]
        u_apply = u_seq[0]

        xk = plant_step(xk, u_apply)
        x_log.append(xk.copy())
        u_log.append(u_apply.astype(np.float32))
        J_log.append(best_J)

        # warm start: shift mean one segment forward
        mean = np.concatenate([mean[1:], mean[-1:]], axis=0)

        if opts.verbose:
            u_str = ", ".join(f"{n}={v:.1f}"
                              for n, v in zip(spec["u_names"], u_apply))
            y_now = xk[spec["tracked_idx"]]
            print(f"  [step {step+1:3d}]  {u_str}  "
                  f"y={np.round(y_now, 3).tolist()}  J={best_J:.3e}  "
                  f"({B}x{iters} rollouts, {ms_l[-1]:.0f} ms)")

    def _mean_tail(v):
        return float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))

    stats = {
        "mpc_optimizer":         "CEM",
        "mpc_horizon":           H,
        "cem_samples":           B,
        "cem_iters":             iters,
        "rollouts_per_step":     B * iters,
        "opt_ms_first_step":     float(ms_l[0]),
        "opt_ms_per_step_mean":  _mean_tail(ms_l),
    }
    return x_log, u_log, J_log, stats


# ══════════════════════════════════════════════════════════════════════
# Closed-loop: model-predictive path integral (batched importance sampling)
# ══════════════════════════════════════════════════════════════════════

def _run_mppi_mpc(learner, problem, spec, x0, opts, t_seg):
    import tensorflow as tf

    DTYPE   = learner.basis.dtype
    H       = max(1, int(opts.horizon or 1))
    n_u     = len(spec["u_lo"])
    n_steps = int(opts.n_steps)
    B       = int(opts.mppi_samples)
    iters   = int(opts.mppi_iters)
    lam     = float(opts.mppi_lambda)
    noise   = float(opts.mppi_noise)

    batch_cost, _ = _make_rollout_fns(learner, H, spec)
    plant_step = _make_plant_step(problem, spec, t_seg)

    u_lo, u_hi = spec["u_lo"], spec["u_hi"]
    u_span     = u_hi - u_lo
    rng        = np.random.default_rng(opts.seed)

    xk    = np.asarray(x0, dtype=np.float32)
    mean  = np.full((H, n_u), 0.5, dtype=np.float64)   # normalized control
    x_log = [xk.copy()]
    u_log: List[np.ndarray] = []
    J_log: List[float] = []
    ms_l  = []

    for step in range(n_steps):
        xk_tf = tf.constant(xk, dtype=DTYPE)
        t0 = time.perf_counter()
        best_J = np.inf
        for _ in range(iters):
            eps = rng.standard_normal((B, H, n_u)) * noise
            Z   = np.clip(mean[None] + eps, 0.0, 1.0)
            J   = batch_cost(xk_tf, tf.constant(Z, dtype=DTYPE)).numpy()
            best_J = min(best_J, float(J.min()))
            # importance weights: softmax of -(J - min)/lambda
            w = np.exp(-(J - J.min()) / max(lam, 1e-8))
            w = w / (w.sum() + 1e-12)
            mean = np.clip(np.einsum("b,bhu->hu", w, Z), 0.0, 1.0)
        ms_l.append((time.perf_counter() - t0) * 1000.0)

        u_seq   = u_lo[None, :] + mean * u_span[None, :]
        u_apply = u_seq[0]
        xk = plant_step(xk, u_apply)
        x_log.append(xk.copy())
        u_log.append(u_apply.astype(np.float32))
        J_log.append(best_J)
        mean = np.concatenate([mean[1:], mean[-1:]], axis=0)   # warm start

        if opts.verbose:
            u_str = ", ".join(f"{n}={v:.1f}"
                              for n, v in zip(spec["u_names"], u_apply))
            y_now = xk[spec["tracked_idx"]]
            print(f"  [step {step+1:3d}]  {u_str}  "
                  f"y={np.round(y_now, 3).tolist()}  J={best_J:.3e}  "
                  f"({B}x{iters} rollouts, {ms_l[-1]:.0f} ms)")

    def _mean_tail(v):
        return float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))

    stats = {
        "mpc_optimizer":        "MPPI",
        "mpc_horizon":          H,
        "mppi_samples":         B,
        "mppi_iters":           iters,
        "rollouts_per_step":    B * iters,
        "opt_ms_first_step":    float(ms_l[0]),
        "opt_ms_per_step_mean": _mean_tail(ms_l),
    }
    return x_log, u_log, J_log, stats


# ══════════════════════════════════════════════════════════════════════
# Economic MPC (fed-batch bioreactor): differentiable + batched
# ══════════════════════════════════════════════════════════════════════
#
# Operator input is 7-D: [Xs, Ss, Ps, Vs, Yx, Sin, inp]. The control is the
# feed rate `inp`; Yx and Sin are held at fixed operating values. The
# economic objective maximises terminal product mass Ps·Vs with penalties on
# feed usage, input roughness, volume overflow, and substrate inhibition —
# the same objective as the legacy loop, expressed as one TF graph so that
# dJ/d(inp sequence) is available by autodiff and B candidate feed profiles
# evaluate in one batched pass.

_BIO_ECON = dict(
    Yx=0.40, Sin=0.8, inp_min=0.005, inp_max=0.200,
    V_max=5.0, S_inh=0.8,
    x_min=(0.0, 0.0, 0.0, 0.0), x_max=(20.0, 5.0, 20.0, 15.0),
    w_terminal=1.0, w_integral=0.1, w_feed=0.01,
    w_smooth=0.1, w_vol=10.0, w_subst=0.5, w_neg=10.0,
)


def _build_economic_spec(opts):
    d = dict(_BIO_ECON)
    if isinstance(opts.control_bounds, (tuple, list)) and len(opts.control_bounds) == 2:
        d["inp_min"], d["inp_max"] = map(float, opts.control_bounds)
    elif isinstance(opts.control_bounds, dict):
        d["inp_min"] = float(opts.control_bounds.get("inp_min", d["inp_min"]))
        d["inp_max"] = float(opts.control_bounds.get("inp_max", d["inp_max"]))
    return d


def _make_economic_rollout_fns(learner, H, spec, t_seg):
    import tensorflow as tf
    DTYPE = learner.basis.dtype

    inp_lo   = tf.constant(spec["inp_min"], dtype=DTYPE)
    inp_span = tf.constant(spec["inp_max"] - spec["inp_min"], dtype=DTYPE)
    Yx  = tf.constant(spec["Yx"],  dtype=DTYPE)
    Sin = tf.constant(spec["Sin"], dtype=DTYPE)
    x_min = tf.constant(spec["x_min"], dtype=DTYPE)
    x_max = tf.constant(spec["x_max"], dtype=DTYPE)
    dt  = tf.constant(float(t_seg), dtype=DTYPE)
    Vmx = tf.constant(spec["V_max"], dtype=DTYPE)
    Sih = tf.constant(spec["S_inh"], dtype=DTYPE)
    wT, wI = tf.constant(spec["w_terminal"], DTYPE), tf.constant(spec["w_integral"], DTYPE)
    wF, wS = tf.constant(spec["w_feed"], DTYPE), tf.constant(spec["w_smooth"], DTYPE)
    wV, wSu, wN = (tf.constant(spec["w_vol"], DTYPE),
                   tf.constant(spec["w_subst"], DTYPE),
                   tf.constant(spec["w_neg"], DTYPE))

    def _rollout_cost(x0_b, Z, inp_prev):
        """x0_b (B,4), Z (B,H,1) in [0,1], inp_prev (B,) → J (B,)"""
        inp = inp_lo + Z[..., 0] * inp_span                 # (B, H)
        xc  = x0_b
        B   = tf.shape(Z)[0]
        J   = tf.zeros(B, dtype=DTYPE)
        prev = inp_prev
        integral = tf.zeros(B, dtype=DTYPE)
        for k in range(H):
            theta = tf.stack([tf.fill([B], Yx), tf.fill([B], Sin), inp[:, k]], axis=1)
            z_in  = tf.concat([xc, theta], axis=1)          # (B, 7)
            out   = learner.forward_segment(z_in, training=False)
            xc    = tf.clip_by_value(out["x_end"], x_min, x_max)
            Ps, Vs, Ss = xc[:, 2], xc[:, 3], xc[:, 1]
            integral = integral + Ps * Vs * dt
            J = (J
                 + wF * inp[:, k] * dt
                 + wS * tf.square(inp[:, k] - prev)
                 + wV * tf.square(tf.nn.relu(Vs - Vmx))
                 + wSu * tf.square(tf.nn.relu(Ss - Sih))
                 + wN * tf.reduce_sum(tf.square(tf.nn.relu(-xc)), axis=1))
            prev = inp[:, k]
        Ps_T, Vs_T = xc[:, 2], xc[:, 3]
        J = J - wT * Ps_T * Vs_T - wI * integral
        return J

    @tf.function(reduce_retracing=True)
    def batch_cost(xk, Z, inp_prev):
        x0_b = tf.repeat(xk[None, :], tf.shape(Z)[0], axis=0)
        ip_b = tf.repeat(tf.reshape(inp_prev, [1]), tf.shape(Z)[0], axis=0)
        return _rollout_cost(x0_b, Z, ip_b)

    @tf.function(reduce_retracing=True)
    def cost_and_grad(xk, z_flat, inp_prev):
        with tf.GradientTape() as tape:
            tape.watch(z_flat)
            Z = tf.reshape(z_flat, (1, H, 1))
            J = _rollout_cost(xk[None, :], Z, tf.reshape(inp_prev, [1]))[0]
        g = tape.gradient(J, z_flat)
        return J, g

    return batch_cost, cost_and_grad


def _run_economic_mpc(learner, problem, x0, opts, t_seg):
    import tensorflow as tf
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize

    spec = _build_economic_spec(opts)
    DTYPE = learner.basis.dtype
    H = max(1, int(opts.horizon or 10))
    n_steps = int(opts.n_steps)
    optimizer = str(getattr(opts, "optimizer", "SLSQP")).upper()
    use_grad = getattr(opts, "gradient", None) is not None
    x_min = np.asarray(spec["x_min"], np.float32)
    x_max = np.asarray(spec["x_max"], np.float32)
    inp_lo, inp_hi = spec["inp_min"], spec["inp_max"]
    inp_span = inp_hi - inp_lo

    batch_cost, cost_and_grad = _make_economic_rollout_fns(learner, H, spec, t_seg)

    def plant_step(xk, inp):
        theta = np.array([spec["Yx"], spec["Sin"], float(inp)], np.float64)
        sol = solve_ivp(lambda t, y: problem.rhs_np(t, y, theta),
                        (0.0, t_seg), np.asarray(xk, np.float64),
                        method="RK45", rtol=1e-6, atol=1e-8)
        return np.clip(sol.y[:, -1].astype(np.float32), x_min, x_max)

    xk = np.asarray(x0, np.float32)
    mean = np.full((H, 1), 0.3, np.float64)
    z0 = np.full(H, 0.3, np.float64)
    inp_prev = 0.04
    rng = np.random.default_rng(opts.seed)
    x_log, u_log, J_log, ms_l = [xk.copy()], [], [], []
    nfev_l, njev_l = [], []

    for step in range(n_steps):
        xk_tf = tf.constant(xk, DTYPE)
        ip_tf = tf.constant(inp_prev, DTYPE)
        t0 = time.perf_counter()

        if optimizer in ("CEM", "MPPI"):
            B = int(opts.cem_samples if optimizer == "CEM" else opts.mppi_samples)
            iters = int(opts.cem_iters if optimizer == "CEM" else opts.mppi_iters)
            std = np.full((H, 1), 0.25, np.float64)
            best = np.inf
            for _ in range(iters):
                Z = np.clip(mean[None] + std[None] * rng.standard_normal((B, H, 1)), 0, 1) \
                    if optimizer == "CEM" else \
                    np.clip(mean[None] + opts.mppi_noise * rng.standard_normal((B, H, 1)), 0, 1)
                J = batch_cost(xk_tf, tf.constant(Z, DTYPE), ip_tf).numpy()
                best = min(best, float(J.min()))
                if optimizer == "CEM":
                    el = Z[np.argsort(J)[:int(opts.cem_elites)]]
                    mean = opts.cem_alpha * el.mean(0) + (1 - opts.cem_alpha) * mean
                    std = np.maximum(opts.cem_alpha * el.std(0) + (1 - opts.cem_alpha) * std, 0.01)
                else:
                    w = np.exp(-(J - J.min()) / max(opts.mppi_lambda, 1e-8)); w /= w.sum() + 1e-12
                    mean = np.clip(np.einsum("b,bhu->hu", w, Z), 0, 1)
            inp_apply = float(inp_lo + mean[0, 0] * inp_span)
            J_log.append(best); nfev_l.append(B * iters); njev_l.append(0)
        else:
            if use_grad and opts.gradient == "autodiff":
                def fun(z):
                    J, g = cost_and_grad(xk_tf, tf.constant(z, DTYPE), ip_tf)
                    return float(J.numpy()), np.asarray(g.numpy(), np.float64)
                res = minimize(fun, z0, jac=True, method="SLSQP",
                               bounds=[(0, 1)] * H, options={"ftol": 1e-8, "maxiter": 200})
            else:
                def fun(z):
                    Z = tf.constant(z.reshape(1, H, 1), DTYPE)
                    return float(batch_cost(xk_tf, Z, ip_tf).numpy()[0])
                res = minimize(fun, z0, method="SLSQP", bounds=[(0, 1)] * H,
                               options={"ftol": 1e-8, "maxiter": 200, "eps": 1e-3})
            z_opt = np.clip(res.x, 0, 1)
            inp_apply = float(inp_lo + z_opt[0] * inp_span)
            J_log.append(float(res.fun)); nfev_l.append(int(res.nfev))
            njev_l.append(int(getattr(res, "njev", 0) or 0))
            z0 = np.concatenate([z_opt[1:], z_opt[-1:]])

        ms_l.append((time.perf_counter() - t0) * 1000.0)
        xk = plant_step(xk, inp_apply)
        x_log.append(xk.copy()); u_log.append(np.array([inp_apply], np.float32))
        mean = np.concatenate([mean[1:], mean[-1:]], axis=0)
        inp_prev = inp_apply
        if opts.verbose:
            print(f"  [step {step+1:3d}]  inp={inp_apply:.4f}  "
                  f"Ps={xk[2]:.3f} Vs={xk[3]:.3f} PsVs={xk[2]*xk[3]:.3f}  "
                  f"J={J_log[-1]:.3e}  {ms_l[-1]:.0f} ms")

    def _tail(v): return float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))
    label = optimizer if optimizer in ("CEM", "MPPI") else f"SLSQP-{opts.gradient or 'fd'}"
    stats = {
        "mpc_optimizer": label, "mpc_horizon": H,
        "opt_nfev_mean": float(np.mean(nfev_l)),
        "opt_njev_mean": float(np.mean(njev_l)),
        "opt_ms_first_step": float(ms_l[0]),
        "opt_ms_per_step_mean": _tail(ms_l),
    }
    if optimizer in ("CEM", "MPPI"):
        stats["rollouts_per_step"] = nfev_l[0]
    return x_log, u_log, J_log, stats


# ══════════════════════════════════════════════════════════════════════
# Entry point (called from workflows/mpc_workflow.run_mpc)
# ══════════════════════════════════════════════════════════════════════

def run_builtin_surrogate_mpc(learner, problem, mpc_name: str, x0, opts, cfg
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, Dict]:
    """Gradient-based or CEM tracking MPC on a trained built-in operator.

    Returns (t, x, u, J, stats).  ``stats`` is merged into the run metadata
    by run_mpc.
    """
    t_seg = float(cfg.DT_SEG)

    # ── Economic MPC (fed-batch bioreactor) ───────────────────────────
    if mpc_name == "bioreactor":
        if opts.verbose:
            optimizer = str(getattr(opts, "optimizer", "SLSQP")).upper()
            mode = (optimizer if optimizer in ("CEM", "MPPI")
                    else f"SLSQP-{(opts.gradient or 'fd')}")
            print(f"[run_mpc/surrogate] economic mode={mode}  "
                  f"horizon={opts.horizon or 10}")
        x_log, u_log, J_log, stats = _run_economic_mpc(
            learner, problem, x0, opts, t_seg)
        time_factor = float(getattr(problem, "time_factor", 1.0))
        t_arr = np.arange(len(x_log), dtype=np.float32) * t_seg * time_factor
        u_arr = np.stack(u_log, axis=0)
        if u_arr.shape[1] == 1:
            u_arr = u_arr[:, 0]
        return (t_arr, np.stack(x_log, axis=0), u_arr.astype(np.float32),
                np.array(J_log, dtype=np.float32), stats)

    spec  = _build_tracking_spec(mpc_name, problem, opts)

    optimizer = str(getattr(opts, "optimizer", "SLSQP")).upper()
    if opts.verbose:
        mode = (optimizer if optimizer in ("CEM", "MPPI")
                else f"SLSQP-{(opts.gradient or 'autodiff')}")
        tracked = [problem.state_labels[i] for i in spec["tracked_idx"]]
        print(f"[run_mpc/surrogate] mode={mode}  horizon={opts.horizon or 1}  "
              f"tracked={tracked}  y_ref={spec['y_ref'].tolist()}")

    if optimizer == "CEM":
        x_log, u_log, J_log, stats = _run_cem_mpc(
            learner, problem, spec, x0, opts, t_seg)
    elif optimizer == "MPPI":
        x_log, u_log, J_log, stats = _run_mppi_mpc(
            learner, problem, spec, x0, opts, t_seg)
    else:
        x_log, u_log, J_log, stats = _run_gradient_mpc(
            learner, problem, spec, x0, opts, t_seg)

    time_factor = float(getattr(problem, "time_factor", 1.0))
    t_arr = np.arange(len(x_log), dtype=np.float32) * t_seg * time_factor

    u_arr = np.stack(u_log, axis=0)
    if u_arr.shape[1] == 1:
        u_arr = u_arr[:, 0]   # match legacy scalar-control shape (n_steps,)

    return (t_arr,
            np.stack(x_log, axis=0),
            u_arr.astype(np.float32),
            np.array(J_log, dtype=np.float32),
            stats)
