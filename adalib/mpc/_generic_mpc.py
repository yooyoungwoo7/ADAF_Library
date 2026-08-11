"""
adalib/mpc/_generic_mpc.py
Generic tracking MPC for user-defined CallableODESystem.

Uses ADA LPA Operator NN as the surrogate prediction model — the same
architecture as the built-in CSTR / triple-tank MPC workflows.

Workflow
--------
1. Validate options (control names, target, dt).
2. Generate (x_0, u) → segment trajectory x(t) training pairs via
   scipy solve_ivp.  Each sample covers one MPC time-step of duration dt.
3. Train OperatorNet: (x_0, u) → W (LPA panel weights, shape state×N_p).
   Loss = trajectory MSE  ||basis(W, x_0)["x"] − x_ref||²
   Extract the trained weights into a pure-numpy LPA surrogate for SLSQP.
4. Run receding-horizon tracking MPC via scipy SLSQP:
   - Prediction: _NumpyLPASurrogate.predict_next (numpy OperatorNet + LPA)
   - True plant: solve_ivp one step per MPC iteration.
5. Save trajectory / controls / metadata and return MPCResult.
"""
from __future__ import annotations

import os
import json
import time
import numpy as np
from typing import List, Optional


# LPA defaults (override via getattr(opts, 'lpa_n_panels', ...) etc.)
_LPA_N_P       = 8    # number of LPA panels
_LPA_MAX_ORDER = 6    # Legendre polynomial order
_LPA_NT_SEG    = 20   # time points per segment during training


# ══════════════════════════════════════════════════════════════════════
# Swish — matches OperatorNet's hidden-layer activation
# ══════════════════════════════════════════════════════════════════════

def _swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ══════════════════════════════════════════════════════════════════════
# Pure-numpy LPA Operator surrogate (for fast SLSQP inference)
# ══════════════════════════════════════════════════════════════════════

class _NumpyLPASurrogate:
    """
    Pure-numpy inference: (x_k, u_k) → x_{k+1} via ADA LPA expansion.

    Step 1 — NN forward: (x_0, u) → W_flat  (numpy Dense layers + Swish)
    Step 2 — LPA endpoint:
        a_0   = W_panel @ mean_vec           (per state)
        A     = coef_mat @ W_panel.T         (M, n_state)
        x_end = x_0 + (a_0*I1_0_end + A.T @ I1_end) / time_scale

    All constants are extracted from the trained TF objects at construction.
    """

    def __init__(self, lw, x_mean, x_std, n_state, N_p,
                 mean_vec, coef_mat, I1_0_end, I1_end, time_scale):
        self.lw = [(np.asarray(W, np.float64), np.asarray(b, np.float64))
                   for W, b in lw]
        self.xm = np.asarray(x_mean,  np.float64)
        self.xs = np.asarray(x_std,   np.float64)
        self.n_state    = int(n_state)
        self.N_p        = int(N_p)
        self.mean_vec   = np.asarray(mean_vec,  np.float64)   # (N_p,)
        self.coef_mat   = np.asarray(coef_mat,  np.float64)   # (M, N_p)
        self.I1_0_end   = float(I1_0_end)
        self.I1_end     = np.asarray(I1_end,    np.float64)   # (M,)
        self.time_scale = float(time_scale)

    # ── Inference ─────────────────────────────────────────────────────

    def _nn_forward(self, z: np.ndarray) -> np.ndarray:
        """(n_state+n_control,) → W_flat (n_state*N_p,)"""
        h = (z - self.xm) / self.xs          # x_std already >= 1e-6 from OperatorNet
        for i, (W_mat, b) in enumerate(self.lw):
            h = h @ W_mat + b
            if i < len(self.lw) - 1:
                h = _swish(h)
        return h

    def _lpa_endpoint(self, W_panel: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """W_panel: (n_state, N_p), x0: (n_state,) → x_end: (n_state,)"""
        a0    = W_panel @ self.mean_vec              # (n_state,)
        A     = self.coef_mat @ W_panel.T            # (M, n_state)
        delta = (a0 * self.I1_0_end + A.T @ self.I1_end) / self.time_scale
        return x0 + delta

    def predict_next(self, xk, uk) -> np.ndarray:
        z       = np.concatenate([np.asarray(xk, float).ravel(),
                                   np.asarray(uk, float).ravel()])
        W_panel = self._nn_forward(z).reshape(self.n_state, self.N_p)
        return self._lpa_endpoint(W_panel, np.asarray(xk, float).ravel())

    # ── Differentiable inference (analytic Jacobian) ──────────────────

    def _endpoint_coef(self) -> np.ndarray:
        """Per-panel LPA endpoint coefficient c (shape (N_p,)) such that
        x_end_j - x0_j = sum_p W[j,p] * c[p]  for every state j.

        Derivation: a0 = W·mean_vec, A = coef_mat·Wᵀ, and
        x_end - x0 = (a0·I1_0_end + Aᵀ·I1_end)/time_scale, all linear in W.
        """
        return ((self.mean_vec * self.I1_0_end
                 + self.coef_mat.T @ self.I1_end)
                / self.time_scale)                     # (N_p,)

    def _nn_forward_jac(self, z: np.ndarray):
        """Return (W_flat, dW_flat/dz) for the MLP with swish hidden units.

        dW_flat/dz has shape (n_state*N_p, n_state+n_control).
        """
        h = (z - self.xm) / self.xs
        # Jacobian of the input normalisation w.r.t. z.
        J = np.diag(1.0 / self.xs)                     # (in, in)
        n_layers = len(self.lw)
        for i, (W_mat, b) in enumerate(self.lw):
            a = h @ W_mat + b                          # pre-activation
            J = W_mat.T @ J                            # linear part
            if i < n_layers - 1:
                s = 1.0 / (1.0 + np.exp(-a))           # sigmoid
                swish_grad = s * (1.0 + a * (1.0 - s))  # d/da [a·sigmoid(a)]
                h = a * s
                J = swish_grad[:, None] * J
            else:
                h = a
        return h, J                                    # (out,), (out, in)

    def predict_next_with_grads(self, xk, uk):
        """Return (x_next, dx_next/dxk, dx_next/duk).

        dx_next/dxk : (n_state, n_state)   dx_next/duk : (n_state, n_control)
        Exact — LPA endpoint is linear in the network output W.
        """
        xk = np.asarray(xk, float).ravel()
        uk = np.asarray(uk, float).ravel()
        z  = np.concatenate([xk, uk])
        n_u = z.shape[0] - self.n_state

        W_flat, dW_dz = self._nn_forward_jac(z)
        W_panel = W_flat.reshape(self.n_state, self.N_p)
        x_next  = self._lpa_endpoint(W_panel, xk)

        c = self._endpoint_coef()                      # (N_p,)
        # d(x_end_j - x0_j)/dz = sum_p c[p] * dW_flat[j*N_p + p]/dz
        dW_dz3 = dW_dz.reshape(self.n_state, self.N_p, -1)
        ddelta_dz = np.einsum("p,jpk->jk", c, dW_dz3)   # (n_state, in)

        dx_dxk = ddelta_dz[:, :self.n_state] + np.eye(self.n_state)  # +x0 term
        dx_duk = ddelta_dz[:, self.n_state:self.n_state + n_u]
        return x_next, dx_dxk, dx_duk

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        d: dict = {
            'n_state':    np.array([self.n_state]),
            'N_p':        np.array([self.N_p]),
            'x_mean':     self.xm,
            'x_std':      self.xs,
            'mean_vec':   self.mean_vec,
            'coef_mat':   self.coef_mat,
            'I1_0_end':   np.array([self.I1_0_end]),
            'I1_end':     self.I1_end,
            'time_scale': np.array([self.time_scale]),
            'n_layers':   np.array([len(self.lw)]),
        }
        for i, (W_mat, b) in enumerate(self.lw):
            d[f'W{i}'] = W_mat
            d[f'b{i}'] = b
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> '_NumpyLPASurrogate':
        d = np.load(path, allow_pickle=False)
        n_state  = int(d['n_state'][0])
        N_p      = int(d['N_p'][0])
        n_layers = int(d['n_layers'][0])
        lw = [(d[f'W{i}'].astype(np.float64), d[f'b{i}'].astype(np.float64))
              for i in range(n_layers)]
        return cls(lw,
                   d['x_mean'].astype(np.float64),   d['x_std'].astype(np.float64),
                   n_state, N_p,
                   d['mean_vec'].astype(np.float64),  d['coef_mat'].astype(np.float64),
                   float(d['I1_0_end'][0]),            d['I1_end'].astype(np.float64),
                   float(d['time_scale'][0]))

    @classmethod
    def from_trained(cls, net, basis, n_state: int) -> '_NumpyLPASurrogate':
        """Extract pure-numpy surrogate from trained OperatorNet + BatchLPABasis."""
        lw = []
        for layer in net.net.layers:
            ws = layer.get_weights()
            if len(ws) == 2:
                lw.append((ws[0].astype(np.float64), ws[1].astype(np.float64)))

        x_mean = net.x_mean.numpy().ravel().astype(np.float64)
        x_std  = net.x_std.numpy().ravel().astype(np.float64)

        N_p        = int(basis.N_p)
        mean_vec   = basis.mean_vec.numpy().astype(np.float64)        # (N_p,)
        coef_mat   = basis.coef_mat.numpy().astype(np.float64)        # (M, N_p)
        time_scale = float(basis.time_scale.numpy())
        I1_0_end   = float(basis.I1_0_cache.numpy()[-1])              # scalar at τ=1
        I1_end     = basis.I1_cache.numpy()[:, -1].astype(np.float64) # (M,) at τ=1

        return cls(lw, x_mean, x_std, n_state, N_p,
                   mean_vec, coef_mat, I1_0_end, I1_end, time_scale)


# ══════════════════════════════════════════════════════════════════════
# Training data generation
# ══════════════════════════════════════════════════════════════════════

def _generate_segments(system, dt: float, Nt: int, n_train: int, n_val: int,
                       state_lo, state_hi, ctrl_lo, ctrl_hi, seed: int = 42):
    """
    Generate (x_0, u) → segment trajectory pairs.

    Each sample: integrate rhs for dt with constant u, recording Nt time
    points (including t=0 and t=dt).  The OperatorNet learns to map
    (x_0, u) → W such that BatchLPABasis(W, x_0)["x"] matches this trajectory.

    Returns
    -------
    X_input_tr : (n_train, n_state+n_control) float32
    X_traj_tr  : (n_train, Nt, n_state) float32
    X_input_va : (n_val,   n_state+n_control) float32
    X_traj_va  : (n_val,   Nt, n_state) float32
    """
    from scipy.integrate import solve_ivp

    n_state   = len(system.state_names)
    n_control = len(ctrl_lo)
    t_eval    = np.linspace(0.0, float(dt), int(Nt))

    def _pairs(n, seed_offset):
        rng    = np.random.RandomState(seed + seed_offset)
        X0     = rng.uniform(state_lo, state_hi, (n, n_state))
        U      = rng.uniform(ctrl_lo,  ctrl_hi,  (n, n_control))
        X_traj = np.zeros((n, Nt, n_state), dtype=np.float32)
        for i in range(n):
            ui  = U[i].tolist()
            sol = solve_ivp(
                lambda t, x: system.rhs(t, x, u=ui),
                (0.0, float(dt)), X0[i].tolist(),
                method='RK45', rtol=1e-6, atol=1e-8,
                t_eval=t_eval, dense_output=False,
            )
            X_traj[i] = sol.y.T.astype(np.float32)
        X_input = np.concatenate([X0, U], axis=1).astype(np.float32)
        return X_input, X_traj

    X_tr, XT_tr = _pairs(n_train, 0)
    X_va, XT_va = _pairs(n_val,   1)
    return X_tr, XT_tr, X_va, XT_va


# ══════════════════════════════════════════════════════════════════════
# LPA Operator training
# ══════════════════════════════════════════════════════════════════════

def _train_lpa_operator(X_input_tr, X_traj_tr, X_input_va, X_traj_va,
                        n_state: int, n_control: int, dt: float,
                        opts) -> _NumpyLPASurrogate:
    """
    Train an LPA Operator NN for a generic CallableODESystem.

    Reuses OperatorNet and BatchLPABasis from the legacy backend, but
    instantiates them with EXPLICIT parameters — independent of any
    registered legacy problem or env-var config state.

    Loss: trajectory MSE  ||basis(net(x_0, u), x_0)["x"] − x_ref||²
    """
    import tensorflow as tf
    from ..utils.legacy_context import _ensure_legacy_path
    _ensure_legacy_path()

    # Import legacy classes; defaults from config don't matter — all params explicit
    from models.operator_net import OperatorNet
    from models.lpa_basis import BatchLPABasis

    tf.get_logger().setLevel('ERROR')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    N_p       = int(getattr(opts, 'lpa_n_panels',  _LPA_N_P))
    max_order = int(getattr(opts, 'lpa_max_order', _LPA_MAX_ORDER))
    Nt        = int(X_traj_tr.shape[1])

    # Input normalisation
    x_mean = X_input_tr.mean(0).astype(np.float32)
    x_std  = np.where(X_input_tr.std(0) > 1e-8,
                       X_input_tr.std(0), 1.0).astype(np.float32)

    # Build OperatorNet and BatchLPABasis with explicit parameters
    net = OperatorNet(
        input_dim=n_state + n_control,
        state_dim=n_state,
        N_p=N_p,
        hidden=int(opts.hidden),
        n_layers=int(opts.n_layers),
        x_mean=x_mean,
        x_std=x_std,
    )
    basis = BatchLPABasis(
        T_seg=float(dt),
        Nt=Nt,
        gamma=1.0,
        max_order=max_order,
        N_p=N_p,
    )
    DTYPE = basis.dtype

    # Cosine LR schedule (matches legacy learner.py)
    n_ep  = int(opts.epochs)
    n_bat = max(1, len(X_input_tr) // int(opts.batch_size))
    sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(opts.lr),
        decay_steps=max(1, n_ep * n_bat),
        alpha=1e-6 / max(float(opts.lr), 1e-12),
    )
    optimizer = tf.keras.optimizers.Adam(sched)

    # TF datasets
    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_input_tr, X_traj_tr))
                .shuffle(len(X_input_tr), seed=42)
                .batch(int(opts.batch_size), drop_remainder=False))
    val_ds = (tf.data.Dataset
              .from_tensor_slices((X_input_va, X_traj_va))
              .batch(max(len(X_input_va), 1)))

    @tf.function
    def _train_step(x_in, x_traj_ref):
        with tf.GradientTape() as tape:
            W    = net(x_in, training=True)                    # (B, n_state, N_p)
            x0   = tf.cast(x_in[:, :n_state], DTYPE)
            out  = basis(W, x0)                                # dict
            loss = tf.reduce_mean(
                tf.square(out["x"] - tf.cast(x_traj_ref, DTYPE)))
        grads, _ = tf.clip_by_global_norm(
            tape.gradient(loss, net.trainable_variables), 1.0)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    @tf.function
    def _val_loss(x_in, x_traj_ref):
        W   = net(x_in, training=False)
        x0  = tf.cast(x_in[:, :n_state], DTYPE)
        out = basis(W, x0)
        return tf.reduce_mean(
            tf.square(out["x"] - tf.cast(x_traj_ref, DTYPE)))

    print_every  = max(1, n_ep // 10)
    best_val     = np.inf
    best_weights = None

    for epoch in range(n_ep):
        tr  = [float(_train_step(xi, xt)) for xi, xt in train_ds]
        val = [float(_val_loss(xi, xt)) for xi, xt in val_ds]
        tr_mean, val_mean = float(np.mean(tr)), float(np.mean(val))

        if val_mean < best_val:
            best_val = val_mean
            best_weights = [v.numpy().copy() for v in net.trainable_variables]

        if opts.verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"  [LPA op {epoch + 1:4d}/{n_ep}]  "
                  f"loss={tr_mean:.4e}  val={val_mean:.4e}")

    # Restore best-validation checkpoint
    if best_weights is not None:
        for var, w in zip(net.trainable_variables, best_weights):
            var.assign(w)

    return _NumpyLPASurrogate.from_trained(net, basis, n_state)


# ══════════════════════════════════════════════════════════════════════
# Physics-only generic Operator training
#
# Used by run_operator() for any CallableODESystem — no reference
# trajectory is ever generated or fit against. BatchLPABasis already
# returns an exact analytic "xdot" alongside "x" (lpa_basis.py), so the
# only change from _train_lpa_operator above is the loss target: the true
# ODE RHS (system.rhs()) instead of a solve_ivp trajectory.
# ══════════════════════════════════════════════════════════════════════

def _sample_inputs(n_train: int, n_val: int, n_state: int, n_control: int,
                   state_lo, state_hi, ctrl_lo, ctrl_hi, seed: int = 42):
    """Sample (x0, u) input pairs only — no reference trajectory.

    Physics-residual training never compares against data, so there is
    nothing to integrate here (unlike _generate_segments above).
    """
    def _pairs(n, seed_offset):
        rng = np.random.RandomState(seed + seed_offset)
        X0 = rng.uniform(state_lo, state_hi, (n, n_state))
        U  = (rng.uniform(ctrl_lo, ctrl_hi, (n, n_control))
              if n_control > 0 else np.zeros((n, 0)))
        return np.concatenate([X0, U], axis=1).astype(np.float32)

    return _pairs(n_train, 0), _pairs(n_val, 1)


def _physics_target(system, x_bt: np.ndarray, u_bt: np.ndarray,
                    n_control: int) -> np.ndarray:
    """True RHS f(x, u) at every (batch, time) point of a segment rollout.

    x_bt : (B, Nt, n_state) numpy — the basis's own reconstructed trajectory
    u_bt : (B, n_control) numpy  — held constant across the Nt segment points
    Returns (B, Nt, n_state) numpy.

    Tries one fully-vectorized call to system.rhs() first (fast — works
    whenever the user's RHS uses plain elementwise numpy ops, since state
    is passed state-major so `a, b, c = x` unpacks into (B*Nt,) arrays).
    Falls back to a per-sample Python loop (always correct, just slower)
    if the vectorized attempt raises — e.g. float(u), math.exp, or builtin
    min()/max() on multi-element arrays all raise cleanly and are caught.
    """
    B, Nt, n_state = x_bt.shape
    x_flat = x_bt.reshape(B * Nt, n_state).astype(np.float64)
    u_flat = (np.repeat(u_bt.astype(np.float64), Nt, axis=0) if n_control > 0
              else np.zeros((B * Nt, 0), dtype=np.float64))

    try:
        x_T = x_flat.T                                    # (n_state, B*Nt)
        u_arg = u_flat.T if n_control > 0 else None        # (n_control, B*Nt)
        out = np.asarray(system.rhs(0.0, x_T, u=u_arg), dtype=np.float64)
        if out.shape != (n_state, B * Nt):
            raise ValueError(
                f"vectorized rhs() returned shape {out.shape}, "
                f"expected {(n_state, B * Nt)}"
            )
        return out.T.reshape(B, Nt, n_state).astype(np.float32)
    except Exception as exc:
        if not _physics_target._warned:
            import warnings
            warnings.warn(
                "[run_operator/generic] system.rhs() is not batch-vectorizable "
                f"({type(exc).__name__}: {exc}) — falling back to a per-sample "
                "Python loop for physics-residual training (slower). For a "
                "faster path, avoid float(u)/math.exp/builtin min()/max() on "
                "arrays inside rhs(); use np.exp/np.minimum/np.maximum instead.",
                stacklevel=2,
            )
            _physics_target._warned = True
        out = np.empty((B * Nt, n_state), dtype=np.float64)
        for k in range(B * Nt):
            u_k = u_flat[k].tolist() if n_control > 0 else None
            out[k] = system.rhs(0.0, x_flat[k], u=u_k)
        return out.reshape(B, Nt, n_state).astype(np.float32)


_physics_target._warned = False


def _train_lpa_operator_physics(X_input_tr, X_input_va, system,
                                n_state: int, n_control: int, dt: float,
                                opts, state_lo, state_hi) -> _NumpyLPASurrogate:
    """
    Train an LPA Operator NN for a generic CallableODESystem — PHYSICS ONLY.

    Same OperatorNet + BatchLPABasis architecture as _train_lpa_operator,
    but the loss compares the basis's own analytic time-derivative
    (BatchLPABasis already returns "xdot") against the true ODE RHS
    evaluated via system.rhs(). No reference trajectory is generated or
    used anywhere in this function.

    Loss: physics residual  mean(((basis(net(x0,u),x0)["xdot"] − f(x,u))
                                   / res_scale)²)
    """
    import tensorflow as tf
    from ..utils.legacy_context import _ensure_legacy_path
    _ensure_legacy_path()

    from models.operator_net import OperatorNet
    from models.lpa_basis import BatchLPABasis

    tf.get_logger().setLevel('ERROR')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    N_p       = int(getattr(opts, 'lpa_n_panels',  _LPA_N_P))
    max_order = int(getattr(opts, 'lpa_max_order', _LPA_MAX_ORDER))
    Nt        = int(getattr(opts, 'lpa_nt_seg',    _LPA_NT_SEG))

    x_mean = X_input_tr.mean(0).astype(np.float32)
    x_std  = np.where(X_input_tr.std(0) > 1e-8,
                       X_input_tr.std(0), 1.0).astype(np.float32)

    net = OperatorNet(
        input_dim=n_state + n_control,
        state_dim=n_state,
        N_p=N_p,
        hidden=int(opts.hidden),
        n_layers=int(opts.n_layers),
        x_mean=x_mean,
        x_std=x_std,
    )
    basis = BatchLPABasis(
        T_seg=float(dt),
        Nt=Nt,
        gamma=1.0,
        max_order=max_order,
        N_p=N_p,
    )
    DTYPE = basis.dtype

    # Per-state residual scale, so states of very different magnitude/speed
    # (e.g. a fast electrical current vs. a slow position) don't dominate
    # the loss purely by scale. Default: state_bounds span / dt — "how much
    # this state could plausibly change over one segment" is a much better
    # proxy for its DERIVATIVE's natural magnitude than the state's own
    # span alone (a state with a short time constant relative to dt has a
    # much larger derivative than its span suggests). Still only a
    # heuristic — for stiff/coupled systems, override per state via
    # opts.res_scale={"state_name": characteristic_dxdt_scale}.
    res_scale_override = getattr(opts, 'res_scale', None)
    if res_scale_override:
        res_scale = np.array([float(res_scale_override.get(n, 1.0))
                              for n in system.state_names], dtype=np.float32)
    else:
        span = np.asarray(state_hi, np.float64) - np.asarray(state_lo, np.float64)
        res_scale = np.where((span > 1e-8) & (span < 1e5),
                              span / max(float(dt), 1e-12), 1.0)
        res_scale = res_scale.astype(np.float32)
    res_scale_tf = tf.constant(res_scale, dtype=DTYPE)   # (n_state,)

    n_ep  = int(opts.epochs)
    n_bat = max(1, len(X_input_tr) // int(opts.batch_size))
    sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(opts.lr),
        decay_steps=max(1, n_ep * n_bat),
        alpha=1e-6 / max(float(opts.lr), 1e-12),
    )
    optimizer = tf.keras.optimizers.Adam(sched)

    train_ds = (tf.data.Dataset
                .from_tensor_slices(X_input_tr)
                .shuffle(len(X_input_tr), seed=42)
                .batch(int(opts.batch_size), drop_remainder=False))
    val_ds = (tf.data.Dataset
              .from_tensor_slices(X_input_va)
              .batch(max(len(X_input_va), 1)))

    def _physics_target_tf(x_traj_f32, u_in_f32):
        target = tf.numpy_function(
            lambda xb, ub: _physics_target(system, xb, ub, n_control),
            [x_traj_f32, u_in_f32], tf.float32,
        )
        target.set_shape(x_traj_f32.shape)
        return tf.cast(target, DTYPE)

    def _step_loss(x_in, training):
        W    = net(x_in, training=training)
        x0   = tf.cast(x_in[:, :n_state], DTYPE)
        u_in = (tf.cast(x_in[:, n_state:], tf.float32) if n_control > 0
                else tf.zeros((tf.shape(x_in)[0], 0), dtype=tf.float32))
        out    = basis(W, x0)
        target = _physics_target_tf(tf.cast(out["x"], tf.float32), u_in)
        resid  = (out["xdot"] - target) / res_scale_tf[None, None, :]
        return tf.reduce_mean(tf.square(resid))

    @tf.function
    def _train_step(x_in):
        with tf.GradientTape() as tape:
            loss = _step_loss(x_in, training=True)
        grads, _ = tf.clip_by_global_norm(
            tape.gradient(loss, net.trainable_variables), 1.0)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    @tf.function
    def _val_loss(x_in):
        return _step_loss(x_in, training=False)

    print_every  = max(1, n_ep // 10)
    best_val     = np.inf
    best_weights = None

    for epoch in range(n_ep):
        tr  = [float(_train_step(xi)) for xi in train_ds]
        val = [float(_val_loss(xi)) for xi in val_ds]
        tr_mean, val_mean = float(np.mean(tr)), float(np.mean(val))

        if val_mean < best_val:
            best_val = val_mean
            best_weights = [v.numpy().copy() for v in net.trainable_variables]

        if opts.verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"  [LPA op-physics {epoch + 1:4d}/{n_ep}]  "
                  f"loss={tr_mean:.4e}  val={val_mean:.4e}")

    if best_weights is not None:
        for var, w in zip(net.trainable_variables, best_weights):
            var.assign(w)

    return _NumpyLPASurrogate.from_trained(net, basis, n_state)


# ══════════════════════════════════════════════════════════════════════
# Receding-horizon MPC loop
# ══════════════════════════════════════════════════════════════════════

def _mpc_loop(surrogate: _NumpyLPASurrogate, system,
              x0, opts,
              ctrl_lo, ctrl_hi,
              tracking_idxs: List[int], y_ref: np.ndarray,
              Q_w: np.ndarray, R_w: np.ndarray, P_w: np.ndarray,
              dt: float):
    """
    Closed-loop receding-horizon tracking MPC.

    At each step:
      1. Optimise [u_0, …, u_{H-1}] via SLSQP using _NumpyLPASurrogate.
      2. Apply u_0 to the TRUE ODE plant (solve_ivp).
      3. Log state, control, cost.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize

    n_control = int(len(ctrl_lo))
    H         = max(1, int(opts.horizon or 1))
    n_steps   = int(opts.n_steps)
    solver    = getattr(opts, 'plant_solver', 'RK45') or 'RK45'

    bounds = [(float(ctrl_lo[j]), float(ctrl_hi[j]))
              for _ in range(H)
              for j in range(n_control)]

    xk     = np.asarray(x0, dtype=np.float64)
    x_log  = [xk.copy()]
    u_log: List[np.ndarray] = []
    J_log: List[float]      = []
    u_prev = np.zeros(n_control, dtype=np.float64)
    t_arr  = np.arange(n_steps + 1, dtype=np.float64) * dt

    grad_mode = (getattr(opts, 'gradient', None) or '').lower()
    tracking_idxs = np.asarray(tracking_idxs, dtype=int)

    def _obj(u_flat, _xk):
        u_seq = u_flat.reshape(H, n_control)
        J  = 0.0
        xi = _xk.copy()
        for k, uk in enumerate(u_seq):
            xi = surrogate.predict_next(xi, uk)
            dy = xi[tracking_idxs] - y_ref
            w  = P_w if (k == H - 1) else Q_w
            J += float(np.dot(w, dy * dy))
            J += float(np.dot(R_w, uk * uk))
        return J

    def _obj_and_grad(u_flat, _xk):
        """Analytic dJ/du by backprop-through-time over the surrogate rollout."""
        u_seq = u_flat.reshape(H, n_control)
        xs, dxk_list, duk_list = [_xk.copy()], [], []
        xi = _xk.copy()
        for uk in u_seq:                              # forward pass
            xi, dxdxk, dxduk = surrogate.predict_next_with_grads(xi, uk)
            xs.append(xi)
            dxk_list.append(dxdxk)
            duk_list.append(dxduk)
        J = 0.0
        gx = np.zeros(len(_xk))                       # adjoint dJ/dx_k
        grad = np.zeros((H, n_control))
        for k in range(H - 1, -1, -1):                # reverse pass
            w  = P_w if (k == H - 1) else Q_w
            dy = xs[k + 1][tracking_idxs] - y_ref
            J += float(np.dot(w, dy * dy)) + float(np.dot(R_w, u_seq[k] ** 2))
            gxk1 = gx.copy()
            gxk1[tracking_idxs] += 2.0 * w * dy       # ∂stage/∂x_{k+1}
            grad[k] = duk_list[k].T @ gxk1 + 2.0 * R_w * u_seq[k]
            gx = dxk_list[k].T @ gxk1                 # propagate to x_k
        return J, grad.reshape(-1)

    for step in range(n_steps):
        xk_now = xk.copy()
        u0_flat = np.tile(u_prev, H).astype(np.float64)
        if grad_mode == 'autodiff':
            res = minimize(
                lambda u: _obj_and_grad(u, xk_now),
                u0_flat, jac=True, method='SLSQP',
                bounds=bounds, options={'ftol': 1e-6, 'maxiter': 300},
            )
        else:
            res = minimize(
                lambda u: _obj(u, xk_now),
                u0_flat, method='SLSQP',
                bounds=bounds, options={'ftol': 1e-6, 'maxiter': 300},
            )
        u_opt = np.clip(res.x[:n_control], ctrl_lo, ctrl_hi)

        # Advance the TRUE ODE plant one step
        _u = u_opt.tolist()
        sol = solve_ivp(
            lambda t, x: system.rhs(t, x, u=_u),
            (0.0, dt), xk.tolist(),
            method=solver, rtol=1e-6, atol=1e-8, dense_output=False,
        )
        xk = sol.y[:, -1].astype(np.float64)

        x_log.append(xk.copy())
        u_log.append(u_opt.copy())
        J_log.append(float(res.fun))
        u_prev = u_opt

        if opts.verbose:
            y_now = xk[tracking_idxs]
            print(f"  [step {step + 1:3d}]  "
                  f"u={np.round(u_opt, 4).tolist()}  "
                  f"y_tracked={np.round(y_now, 4).tolist()}  "
                  f"J={res.fun:.3e}")

    return (
        t_arr.astype(np.float32),
        np.stack(x_log, axis=0).astype(np.float32),
        np.stack(u_log, axis=0).astype(np.float32),
        np.array(J_log, dtype=np.float32),
    )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _write_meta_json(work_dir: str, metadata: dict) -> None:
    meta_path = os.path.join(work_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(
            {k: (v if isinstance(v, (int, float, str, bool, type(None), list))
                 else str(v))
             for k, v in metadata.items()},
            f, indent=2,
        )


def _build_meta(system_name: str, opts, n_steps: int, dt: float,
                elapsed: float, N_p: int, max_order: int, Nt: int) -> dict:
    return {
        'system_name':      system_name,
        'mpc_problem_name': 'generic_callable',
        'surrogate':        'lpa_operator_nn',
        'mode':             getattr(opts, 'mode', 'tracking'),
        'basis':            'lpa',
        'lpa_n_panels':     N_p,
        'lpa_max_order':    max_order,
        'lpa_nt_seg':       Nt,
        'n_steps':          n_steps,
        'horizon':          getattr(opts, 'horizon', 1) or 1,
        'dt':               float(dt),
        'n_train':          int(opts.n_train),
        'epochs':           int(opts.epochs),
        'hidden':           int(opts.hidden),
        'n_layers':         int(opts.n_layers),
        'lr':               float(opts.lr),
        'elapsed_s':        float(elapsed),
        'ms_per_step':      float(elapsed / n_steps * 1000) if n_steps > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
# Public entry point  (called from mpc_workflow.run_mpc)
# ══════════════════════════════════════════════════════════════════════

def run_generic_mpc(system, x0, t_span, opts):
    """
    Generic tracking MPC for a :class:`CallableODESystem`.

    Uses an ADA LPA Operator NN as the surrogate — the same OperatorNet /
    BatchLPABasis architecture used for the built-in CSTR and triple-tank
    systems, instantiated with problem-independent explicit parameters.

    Parameters
    ----------
    system : CallableODESystem
    x0     : array-like — initial plant state
    t_span : (t0, t1) or None
    opts   : MPCOptions (already resolved by run_mpc)

    Returns
    -------
    MPCResult
    """
    from ..workflows.mpc_workflow import MPCResult

    # ── 1. Resolve control / tracking meta ────────────────────────────
    ctrl_names = (getattr(opts, 'control_inputs', None) or
                  list(system.control_names or []))
    if not ctrl_names:
        raise ValueError(
            "system.control_names is empty and opts.control_inputs is not set. "
            "Provide control_names when constructing CallableODESystem, "
            "or set opts.control_inputs=['u1', ...]."
        )

    tracked_names = (getattr(opts, 'controlled_variables', None) or
                     list(system.state_names))

    target = opts.target or {}
    if not target:
        raise ValueError(
            "opts.target must specify reference values, "
            "e.g. {'x': 1.0} or [1.0, 0.0]."
        )

    # ── 2. Resolve dt ─────────────────────────────────────────────────
    dt = getattr(opts, 'dt', None)
    if dt is None and t_span is not None:
        dt = (float(t_span[1]) - float(t_span[0])) / max(1, int(opts.n_steps))
    if dt is None or dt <= 0:
        raise ValueError(
            "opts.dt must be a positive float (segment duration). "
            "Alternatively supply t_span so dt = (t1-t0)/n_steps."
        )

    n_state   = len(system.state_names)
    n_control = len(ctrl_names)

    # ── 3. Resolve bounds ──────────────────────────────────────────────
    def _bound_arrays(names, bound_dict):
        lo, hi = [], []
        for name in names:
            bnd = bound_dict.get(name, (None, None))
            lo.append(-1e6 if bnd[0] is None else float(bnd[0]))
            hi.append( 1e6 if bnd[1] is None else float(bnd[1]))
        return np.array(lo, np.float64), np.array(hi, np.float64)

    ctrl_b  = getattr(opts, 'control_bounds', None) or system.control_bounds or {}
    state_b = getattr(opts, 'state_bounds',   None) or system.state_bounds   or {}

    ctrl_lo,  ctrl_hi  = _bound_arrays(ctrl_names,         ctrl_b)
    state_lo, state_hi = _bound_arrays(system.state_names, state_b)

    for i in range(n_state):
        if state_lo[i] < -1e5: state_lo[i] = -10.0
        if state_hi[i] >  1e5: state_hi[i] =  10.0

    # ── 4. Tracking targets and weight vectors ─────────────────────────
    state_idx     = {n: i for i, n in enumerate(system.state_names)}
    tracking_idxs = [state_idx[n] for n in tracked_names if n in state_idx]
    n_tracked     = len(tracking_idxs)

    if isinstance(target, dict):
        y_ref = np.array([float(target.get(n, 0.0)) for n in tracked_names],
                         dtype=np.float64)
    else:
        y_ref = np.asarray(target, dtype=np.float64)

    def _w(attr, default):
        v = getattr(opts, attr, None)
        return np.asarray(v if v is not None else default, np.float64).ravel()

    Q_w = _w('tracking_weights', np.ones(n_tracked))
    R_w = _w('control_weights',  np.ones(n_control) * 0.01)
    P_w = _w('terminal_weights', Q_w)
    if len(Q_w) != n_tracked: Q_w = np.ones(n_tracked,  np.float64)
    if len(R_w) != n_control: R_w = np.ones(n_control, np.float64) * 0.01
    if len(P_w) != n_tracked: P_w = Q_w.copy()

    # ── 5. LPA operator hyperparameters ───────────────────────────────
    N_p       = int(getattr(opts, 'lpa_n_panels',  _LPA_N_P))
    max_order = int(getattr(opts, 'lpa_max_order', _LPA_MAX_ORDER))
    Nt        = int(getattr(opts, 'lpa_nt_seg',    _LPA_NT_SEG))

    # ── 6. Work directories ────────────────────────────────────────────
    work_dir    = opts.work_dir or f'./runs/{system.name}_generic_mpc'
    work_dir    = os.path.abspath(work_dir)
    op_dir      = getattr(opts, 'operator_work_dir', None) or os.path.join(work_dir, 'operator')
    mpc_res_dir = getattr(opts, 'mpc_result_dir',    None) or os.path.join(work_dir, 'mpc')
    data_dir    = getattr(opts, 'data_dir',           None) or os.path.join(op_dir, 'data')
    ckpt_dir    = getattr(opts, 'checkpoint_dir',     None) or os.path.join(op_dir, 'checkpoints')
    log_dir     = getattr(opts, 'log_dir',            None) or os.path.join(op_dir, 'logs')

    for d in [work_dir, op_dir, mpc_res_dir, data_dir, ckpt_dir, log_dir,
              os.path.join(mpc_res_dir, 'trajectories'),
              os.path.join(mpc_res_dir, 'controls'),
              os.path.join(mpc_res_dir, 'costs')]:
        os.makedirs(d, exist_ok=True)

    paths = {
        'work_dir':          work_dir,
        'operator_work_dir': op_dir,
        'mpc_result_dir':    mpc_res_dir,
        'operator_dir':      op_dir,
        'mpc_dir':           mpc_res_dir,
        'data_dir':          data_dir,
        'checkpoint_dir':    ckpt_dir,
        'log_dir':           log_dir,
        'trajectory_path':   None,
        'control_path':      None,
        'cost_path':         None,
    }

    surr_path = os.path.join(ckpt_dir, 'lpa_operator.npz')
    data_path = os.path.join(data_dir, 'generic_data.npz')

    if opts.verbose:
        print(f"[run_mpc/generic] system={system.name}  "
              f"controls={ctrl_names}  tracked={tracked_names}")
        print(f"[run_mpc/generic] dt={dt:.4f}  n_steps={opts.n_steps}  "
              f"horizon={opts.horizon or 1}  target={target}")
        print(f"[run_mpc/generic] LPA: N_p={N_p}  max_order={max_order}  Nt_seg={Nt}")
        print(f"[run_mpc/generic] work_dir={work_dir}")

    # ── 7. Data generation ─────────────────────────────────────────────
    op_result_info: dict = {}
    X_input_tr = X_traj_tr = X_input_va = X_traj_va = None

    generate = opts.generate_data and not opts.reuse_existing_data

    if generate or getattr(opts, 'force_rebuild_data', False):
        if opts.verbose:
            print(f"[run_mpc/generic] Generating {opts.n_train} train / "
                  f"{opts.n_val} val segment trajectories …")
        X_input_tr, X_traj_tr, X_input_va, X_traj_va = _generate_segments(
            system, dt, Nt, opts.n_train, opts.n_val,
            state_lo, state_hi, ctrl_lo, ctrl_hi, seed=opts.seed,
        )
        np.savez_compressed(data_path,
                            X_input_tr=X_input_tr, X_traj_tr=X_traj_tr,
                            X_input_va=X_input_va, X_traj_va=X_traj_va)
        if opts.verbose:
            print(f"[run_mpc/generic] Data saved → {data_path}")

    elif opts.reuse_existing_data:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"reuse_existing_data=True but data not found:\n  {data_path}\n"
                "Set generate_data=True to create it."
            )
        d = np.load(data_path)
        try:
            X_input_tr = d['X_input_tr']
            X_traj_tr  = d['X_traj_tr']
            X_input_va = d['X_input_va']
            X_traj_va  = d['X_traj_va']
        except KeyError:
            raise ValueError(
                f"Data file {data_path!r} uses an old format. "
                "Regenerate with generate_data=True, reuse_existing_data=False."
            ) from None
        if opts.verbose:
            print(f"[run_mpc/generic] Loaded data: "
                  f"{X_input_tr.shape[0]} train segments")
    else:
        if os.path.exists(data_path):
            d = np.load(data_path)
            try:
                X_input_tr = d['X_input_tr']
                X_traj_tr  = d['X_traj_tr']
                X_input_va = d['X_input_va']
                X_traj_va  = d['X_traj_va']
            except KeyError:
                pass  # old format — will error explicitly at training step

    # ── 8. Operator training ───────────────────────────────────────────
    surrogate: Optional[_NumpyLPASurrogate] = None

    if opts.train_operator and not opts.reuse_existing_operator:
        if X_input_tr is None:
            raise RuntimeError(
                "train_operator=True but no training data available. "
                "Set generate_data=True or reuse_existing_data=True."
            )
        if opts.verbose:
            print(f"[run_mpc/generic] Training LPA Operator NN  "
                  f"epochs={opts.epochs}  hidden={opts.hidden}  "
                  f"n_layers={opts.n_layers}  N_p={N_p}  max_order={max_order}")
        t0 = time.perf_counter()
        surrogate = _train_lpa_operator(
            X_input_tr, X_traj_tr, X_input_va, X_traj_va,
            n_state, n_control, dt, opts,
        )
        surrogate.save(surr_path)
        elapsed_tr = time.perf_counter() - t0
        op_result_info = {
            'train_info':      {'elapsed_s': elapsed_tr},
            'best_checkpoint': surr_path,
            'best_val_loss':   None,
        }
        if opts.verbose:
            print(f"[run_mpc/generic] LPA operator saved → {surr_path}")

    elif opts.reuse_existing_operator:
        if not os.path.exists(surr_path):
            raise FileNotFoundError(
                f"reuse_existing_operator=True but LPA operator not found:\n"
                f"  {surr_path}\n"
                "Set reuse_existing_operator=False to retrain."
            )
        surrogate = _NumpyLPASurrogate.load(surr_path)
        op_result_info = {'best_checkpoint': surr_path, 'reused': True}
        if opts.verbose:
            print(f"[run_mpc/generic] Loaded LPA operator ← {surr_path}")

    if surrogate is None:
        raise RuntimeError(
            "No surrogate available. "
            "Set train_operator=True or reuse_existing_operator=True."
        )

    # ── 9. Closed-loop MPC ─────────────────────────────────────────────
    if not opts.run_closed_loop:
        metadata = _build_meta(system.name, opts, opts.n_steps, dt,
                                0.0, N_p, max_order, Nt)
        _write_meta_json(work_dir, metadata)
        return MPCResult(
            t=np.array([], np.float32), x=np.array([], np.float32),
            u=np.array([], np.float32), cost=np.array([], np.float32),
            operator_result=op_result_info,
            metadata=metadata,
            paths=paths,
            system=system,
        )

    if opts.verbose:
        print(f"\n[run_mpc/generic] Closed-loop MPC  n_steps={opts.n_steps}")

    t0_wall = time.perf_counter()
    t_arr, x_arr, u_arr, J_arr = _mpc_loop(
        surrogate, system, x0, opts,
        ctrl_lo, ctrl_hi, tracking_idxs, y_ref,
        Q_w, R_w, P_w, dt,
    )
    elapsed = time.perf_counter() - t0_wall

    # ── 10. Save results ───────────────────────────────────────────────
    traj_path = os.path.join(mpc_res_dir, 'trajectories', 'trajectory.npz')
    ctrl_path = os.path.join(mpc_res_dir, 'controls',     'controls.npz')
    cost_path = os.path.join(mpc_res_dir, 'costs',        'costs.npz')
    cl_path   = os.path.join(mpc_res_dir, 'closed_loop.npz')

    x0_arr = np.asarray(x0, dtype=np.float32)
    np.savez_compressed(traj_path, t=t_arr, x=x_arr, x0=x0_arr)
    np.savez_compressed(ctrl_path, u=u_arr)
    np.savez_compressed(cost_path, cost=J_arr)
    np.savez_compressed(cl_path,   t=t_arr, x=x_arr, u=u_arr, cost=J_arr, x0=x0_arr)

    paths['trajectory_path']  = traj_path
    paths['control_path']     = ctrl_path
    paths['cost_path']        = cost_path
    paths['closed_loop_path'] = cl_path

    if opts.verbose:
        print(f"\n[run_mpc/generic] Completed {opts.n_steps} steps in "
              f"{elapsed:.2f}s ({elapsed / opts.n_steps * 1000:.1f} ms/step)")
        print(f"[run_mpc/generic] Results → {mpc_res_dir}")

    # ── 11. Metadata ───────────────────────────────────────────────────
    metadata = _build_meta(system.name, opts, opts.n_steps, dt,
                            elapsed, N_p, max_order, Nt)
    metadata['final_state'] = x_arr[-1].tolist() if len(x_arr) > 0 else None
    _write_meta_json(work_dir, metadata)

    return MPCResult(
        t=t_arr,
        x=x_arr,
        u=u_arr,
        cost=J_arr,
        operator_result=op_result_info,
        metadata=metadata,
        paths=paths,
        system=system,
    )
