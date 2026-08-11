"""
adalib/inverse/solver.py

InverseSolver — piecewise ADA-seq inverse training.

Default optimization (joint):

    Phase A  — optional data pre-fit:
        W only, loss = L_data.  Warm-starts ADA trajectory before parameters
        are allowed to move.  Runs on the first pass only.

    Phase B  — joint Adam:
        Every step: grad(L_total, W + θ) → optimizer.apply_gradients

    Phase C  — joint L-BFGS:
        Flatten [W, θ], minimize L_total jointly.

Loss:
    L_total  = lambda_physics * L_physics + lambda_data * L_data
    L_physics = (1/n_state) Σ_j mean(res_j²)        [optionally scaled]
    L_data    = (1/n_obs_st) Σ_j mean((ŷ_j-y_j^obs)²)  [optionally scaled]

Experimental strategy (alternating):
    W-only Adam (phys+data) → θ-only Adam (phys) → W-only L-BFGS → θ-only L-BFGS
    Retained for comparison; not the default.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize
import tensorflow as tf

from ..utils.paths import legacy_forward_root as _legacy_forward_root

_FWD_PATH = str(_legacy_forward_root())
if _FWD_PATH not in sys.path:
    sys.path.insert(0, _FWD_PATH)

from pinn_lib.ADAF_seq.core.model_adaf import ADAF_Reusable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tf_dtype(dtype_str: str) -> tf.DType:
    d = dtype_str.lower()
    if d in ("float64", "fp64", "double"):
        return tf.float64
    return tf.float32


def _eval_g2_at_x(model: ADAF_Reusable,
                  a0, a_n, b_n,
                  x_obs: tf.Tensor) -> tf.Tensor:
    """Evaluate ADAF g2 at arbitrary normalized x coordinates.

    Differentiable w.r.t. W via (a0, a_n, b_n).
    """
    f  = model.f                          # (N_m, 1)
    f2 = model.f2                         # (N_m, 1)

    xo = tf.reshape(x_obs, (1, -1))       # (1, N_obs)
    xo_over_f = xo / f                   # (N_m, N_obs)
    S = tf.sin(xo_over_f)                 # (N_m, N_obs)
    C = tf.cos(xo_over_f)                 # (N_m, N_obs)

    term1 = 0.5 * (-f * b_n[:, None]) * (f * S - xo)   # (N_m, N_obs)
    term2 = 0.5 * (f2 * a_n[:, None]) * (1.0 - C)       # (N_m, N_obs)

    g2 = (
        (a0 / 4.0) * tf.square(x_obs)
        + tf.reduce_sum(term1 + term2, axis=0)
        + model.init1 * x_obs
        + model.init2
    )
    return g2


def _build_p_list(params_resolved: List[Any]) -> List[Any]:
    """Build the p list for rhs_tf, substituting .constrained for InverseParameter."""
    from .parameter import InverseParameter
    result = []
    for v in params_resolved:
        if isinstance(v, InverseParameter):
            result.append(v.constrained)
        else:
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# InverseSolver
# ---------------------------------------------------------------------------

class InverseSolver:
    """Piecewise ADA-F inverse solver with joint parameter optimization.

    Do not instantiate directly; use :func:`run_inverse` instead.
    """

    def __init__(self, system, basis: str = "adaf"):
        if basis.lower() != "adaf":
            raise NotImplementedError(
                "InverseSolver currently supports basis='adaf' only."
            )
        self.system = system
        self.basis = basis.lower()

    # ------------------------------------------------------------------
    def solve(
        self,
        x0:               List[float],
        t_span:           Tuple[float, float],
        params_resolved:  List[Any],
        inverse_params:   Dict[str, Any],
        data,
        *,
        n_seg:    int   = 20,
        N_p:      int   = 5,
        N_m:      int   = 100,
        Nt_total: int   = 1000,
        gamma:    float = 0.8,
        L:        float = 1.0,
        lambda_physics: float = 1.0,
        lambda_data:    float = 10.0,
        epochs:     int   = 5,
        adam_inner: int   = 200,
        adam_lr:    float = 1e-3,
        use_lbfgs:  bool  = True,
        n_passes:   int   = 1,
        dtype:      str   = "float64",
        verbose:    bool  = True,
        log_every:  int   = 0,
        param_log_every: int = 1,
        training_strategy:      str  = "joint",
        data_prefit_steps:      int  = 0,
        normalize_data_loss:    bool = True,
        normalize_physics_loss: bool = False,
        warm_seg_passes:        int  = 1,
        n_warm_segs:            int  = 3,
    ) -> Dict:
        """Run piecewise inverse training.

        Returns
        -------
        dict with keys:
            trajectory           : np.ndarray  (Nt_total, n_state)
            t                    : np.ndarray  (Nt_total,)
            loss_history         : list of float  — one entry per optimizer step
            physics_loss_history : list of float
            data_loss_history    : list of float
            param_history        : dict {name: list of float}
            runtime_sec          : float
        """
        from .parameter import InverseParameter

        lb = float(t_span[0])
        ub = float(t_span[1])
        n_state = len(x0)
        DTYPE = _to_tf_dtype(dtype)

        strategy = training_strategy.lower()
        if strategy not in ("joint", "alternating"):
            raise ValueError(
                f"training_strategy must be 'joint' or 'alternating', got {strategy!r}"
            )

        # ── Validate Nt_total divisibility ────────────────────────────
        if Nt_total % n_seg != 0:
            Nt_total = (Nt_total // n_seg) * n_seg
            if verbose:
                print(f"[InverseSolver] Nt_total adjusted to {Nt_total} "
                      f"(divisible by n_seg={n_seg}).")

        Nt_seg = Nt_total // n_seg
        dt_seg = (ub - lb) / n_seg
        s1_val = L * gamma / dt_seg
        s1 = tf.constant(s1_val, dtype=DTYPE)

        x_norm_np = np.linspace(0.0, L * gamma, Nt_seg).astype(
            DTYPE.as_numpy_dtype
        )
        x_norm = tf.constant(x_norm_np, dtype=DTYPE)

        # ── Build ADAF models ─────────────────────────────────────────
        models = [
            ADAF_Reusable(
                x_norm,
                init1=0.0,
                init2=float(x0[j]),
                init3=0.0,
                N_p=N_p,
                N_m=N_m,
                L=L,
                gamma=gamma,
                dtype=DTYPE,
                name=f"InvADAF_{j}",
            )
            for j in range(n_state)
        ]

        # ── Collect trainable variables ───────────────────────────────
        inv_vars = []
        for name_k, ip in inverse_params.items():
            if isinstance(ip, InverseParameter):
                if ip._raw_var is None:
                    ip.build(dtype=DTYPE, name=name_k)
                inv_vars.append(ip._raw_var)

        W_vars        = [m.W for m in models]
        all_trainable = W_vars + inv_vars

        # ── Build optimizers ──────────────────────────────────────────
        if strategy == "joint":
            joint_optimizer  = tf.keras.optimizers.Adam(learning_rate=adam_lr)
            prefit_optimizer = tf.keras.optimizers.Adam(learning_rate=adam_lr)
            joint_optimizer.build(all_trainable)
            prefit_optimizer.build(W_vars)
        else:
            W_optimizer     = tf.keras.optimizers.Adam(learning_rate=adam_lr)
            theta_optimizer = tf.keras.optimizers.Adam(learning_rate=adam_lr)
            W_optimizer.build(W_vars)
            if inv_vars:
                theta_optimizer.build(inv_vars)

        # ── Storage ───────────────────────────────────────────────────
        t_full = np.linspace(lb, ub, Nt_total).astype(DTYPE.as_numpy_dtype)
        trajectory = np.zeros((Nt_total, n_state), dtype=DTYPE.as_numpy_dtype)

        loss_history:         List[float] = []
        physics_loss_history: List[float] = []
        data_loss_history:    List[float] = []
        param_history = {nm: [] for nm in inverse_params}

        # ── L-BFGS options ────────────────────────────────────────────
        lp_options = {
            "maxiter": 200,
            "maxfun":  20000,
            "maxcor":  50,
            "maxls":   50,
            "ftol":    np.finfo(float).eps,
            "gtol":    np.finfo(float).eps,
            "iprint":  -1,
        }
        lp_options_theta = {
            "maxiter": 50,
            "maxfun":  5000,
            "maxcor":  20,
            "maxls":   20,
            "ftol":    1e-8,
            "gtol":    1e-6,
            "iprint":  -1,
        }

        obs_t  = data.t
        obs_y  = data.y
        obs_si = data.state_indices

        segment_W_cache = [[None] * n_state for _ in range(n_seg)]
        zero3 = tf.constant(0.0, dtype=DTYPE)

        t_all0 = time.perf_counter()
        total_adam_steps = 0

        _total_passes = n_passes + max(0, warm_seg_passes - 1)
        for pass_idx in range(_total_passes):
            _is_warm = pass_idx >= n_passes
            _seg_end = min(n_warm_segs, n_seg) if _is_warm else n_seg
            init1  = [tf.constant(0.0,          dtype=DTYPE)] * n_state
            init2  = [tf.constant(float(x0[j]), dtype=DTYPE) for j in range(n_state)]
            prev_W = [None] * n_state

            if verbose and _total_passes > 1:
                if _is_warm:
                    _wi = pass_idx - n_passes + 1
                    print(f"\n[InverseSolver] === Warm pass {_wi}/{warm_seg_passes-1} "
                          f"(segs 0–{_seg_end-1}) ===")
                else:
                    print(f"\n[InverseSolver] === Pass {pass_idx + 1}/{n_passes} ===")

            for k in range(_seg_end):
                t_k  = lb + k * dt_seg
                t_k1 = lb + (k + 1) * dt_seg
                i0 = k * Nt_seg
                i1 = (k + 1) * Nt_seg

                # ── Set ICs ──────────────────────────────────────────
                for j in range(n_state):
                    models[j].set_inits(init1[j], init2[j], zero3)

                # ── Warm-start W ──────────────────────────────────────
                for j in range(n_state):
                    if pass_idx > 0 and segment_W_cache[k][j] is not None:
                        models[j].W.assign(segment_W_cache[k][j])
                    elif prev_W[j] is not None:
                        models[j].W.assign(prev_W[j])

                # ── Find observations in this segment ─────────────────
                if k < n_seg - 1:
                    mask = (obs_t >= t_k - 1e-12) & (obs_t < t_k1 - 1e-12)
                else:
                    mask = (obs_t >= t_k - 1e-12) & (obs_t <= t_k1 + 1e-12)

                has_obs    = bool(np.any(mask))
                x_obs_tf   = None
                y_obs_list = None
                data_scales = None

                if has_obs:
                    t_obs_seg = obs_t[mask]
                    y_obs_seg = obs_y[mask]
                    t_local   = t_obs_seg - t_k
                    x_obs_np  = (t_local * L * gamma / dt_seg).astype(
                        DTYPE.as_numpy_dtype
                    )
                    x_obs_tf  = tf.constant(x_obs_np, dtype=DTYPE)
                    y_obs_list = [
                        tf.constant(
                            y_obs_seg[:, col].astype(DTYPE.as_numpy_dtype),
                            dtype=DTYPE,
                        )
                        for col in range(len(obs_si))
                    ]

                    # Fixed data scales (per observed state)
                    if normalize_data_loss:
                        data_scales = []
                        for col in range(len(obs_si)):
                            rms = float(
                                np.sqrt(np.mean(y_obs_seg[:, col] ** 2)) + 1e-8
                            )
                            data_scales.append(tf.constant(rms, dtype=DTYPE))

                # Fixed physics scales
                phys_scales = None
                if normalize_physics_loss:
                    phys_scales = []
                    for j in range(n_state):
                        mag = float(abs(x0[j])) if abs(x0[j]) > 1e-8 else 1.0
                        phys_scales.append(tf.constant(mag * s1_val, dtype=DTYPE))

                # =============================================================
                # Optimization phases
                # =============================================================
                if strategy == "joint":

                    # Phase A: data pre-fit (first pass only, W only, L_data)
                    if data_prefit_steps > 0 and pass_idx == 0 and has_obs:
                        _prefit_fn = self._make_compiled_prefit_step(
                            models, W_vars, prefit_optimizer,
                            obs_si, x_obs_tf, y_obs_list, DTYPE,
                            data_scales=data_scales,
                        )
                        for _ in range(data_prefit_steps):
                            _prefit_fn()

                    # Phase B: joint Adam (W + θ, L_total)
                    _step_fn = self._make_compiled_joint_step(
                        models, all_trainable, n_state, self.system,
                        params_resolved, joint_optimizer, s1,
                        has_obs, x_obs_tf, y_obs_list, obs_si,
                        lambda_physics, lambda_data, DTYPE,
                        data_scales=data_scales, phys_scales=phys_scales,
                    )
                    total_joint_steps = epochs * adam_inner
                    for step_i in range(total_joint_steps):
                        phys_l, data_l, total_l = _step_fn()
                        total_adam_steps += 1
                        pf = float(phys_l)
                        df = float(data_l)
                        tf_ = float(total_l)
                        loss_history.append(tf_)
                        physics_loss_history.append(pf)
                        data_loss_history.append(df)

                        if param_log_every > 0 and (total_adam_steps % param_log_every == 0):
                            for nm, ip in inverse_params.items():
                                if isinstance(ip, InverseParameter):
                                    param_history[nm].append(ip.numpy_value)
                                else:
                                    param_history[nm].append(float(ip))

                        if verbose and log_every > 0 and (step_i + 1) % log_every == 0:
                            print(
                                f"  [step {total_adam_steps}] "
                                f"total={tf_:.3e}  phys={pf:.3e}  data={df:.3e}"
                            )

                    # Phase C: joint L-BFGS (W + θ, L_total)
                    if use_lbfgs:
                        lbfgs_x0 = self._lbfgs_pack(all_trainable)
                        scipy.optimize.minimize(
                            fun=self._make_lbfgs_objective(
                                models, all_trainable, n_state, self.system,
                                params_resolved, inv_vars,
                                s1, has_obs,
                                x_obs_tf if has_obs else None,
                                y_obs_list if has_obs else None,
                                obs_si, lambda_physics, lambda_data, DTYPE,
                                data_scales=data_scales, phys_scales=phys_scales,
                            ),
                            x0=lbfgs_x0,
                            jac=True,
                            method="L-BFGS-B",
                            options=lp_options,
                        )
                        # Record one entry for the L-BFGS result
                        phys_l, data_l, _ = self._compute_losses(
                            models, n_state, self.system, params_resolved, inv_vars,
                            s1, has_obs,
                            x_obs_tf if has_obs else None,
                            y_obs_list if has_obs else None,
                            obs_si, DTYPE,
                            data_scales=data_scales, phys_scales=phys_scales,
                        )
                        total_l = lambda_physics * float(phys_l) + lambda_data * float(data_l)
                        loss_history.append(total_l)
                        physics_loss_history.append(float(phys_l))
                        data_loss_history.append(float(data_l))
                        for nm, ip in inverse_params.items():
                            if isinstance(ip, InverseParameter):
                                param_history[nm].append(ip.numpy_value)
                            else:
                                param_history[nm].append(float(ip))

                else:  # "alternating"
                    last_total = 0.0
                    last_phys  = 0.0
                    last_data  = 0.0

                    for ep in range(epochs):
                        # Stage 1: W step (phys + data, θ frozen)
                        for inner in range(adam_inner):
                            phys_l, data_l, total_l = self._W_step(
                                models, W_vars, W_optimizer,
                                n_state, self.system, params_resolved, inv_vars,
                                s1, has_obs,
                                x_obs_tf if has_obs else None,
                                y_obs_list if has_obs else None,
                                obs_si, lambda_physics, lambda_data, DTYPE,
                            )
                            last_total = float(total_l)
                            last_phys  = float(phys_l)
                            last_data  = float(data_l)

                        # Stage 2: θ step (phys only, W frozen)
                        if inv_vars:
                            for inner in range(adam_inner):
                                phys_l = self._theta_step(
                                    models, n_state, self.system,
                                    params_resolved, inv_vars,
                                    theta_optimizer, s1, DTYPE,
                                )
                                last_phys  = float(phys_l)
                                last_total = lambda_physics * last_phys + lambda_data * last_data

                                total_adam_steps += 1
                                if param_log_every > 0 and (total_adam_steps % param_log_every == 0):
                                    for nm, ip in inverse_params.items():
                                        if isinstance(ip, InverseParameter):
                                            param_history[nm].append(ip.numpy_value)
                                        else:
                                            param_history[nm].append(float(ip))

                    # one entry per segment for alternating mode
                    loss_history.append(last_total)
                    physics_loss_history.append(last_phys)
                    data_loss_history.append(last_data)

                    # Stage A/B: alternating L-BFGS
                    if use_lbfgs:
                        lbfgs_x0_W = self._lbfgs_pack(W_vars)
                        scipy.optimize.minimize(
                            fun=self._make_lbfgs_W_objective(
                                models, W_vars, n_state, self.system, params_resolved,
                                s1, has_obs,
                                x_obs_tf if has_obs else None,
                                y_obs_list if has_obs else None,
                                obs_si, lambda_physics, lambda_data, DTYPE,
                            ),
                            x0=lbfgs_x0_W, jac=True, method="L-BFGS-B",
                            options=lp_options,
                        )
                        if inv_vars:
                            lbfgs_x0_theta = self._lbfgs_pack(inv_vars)
                            scipy.optimize.minimize(
                                fun=self._make_lbfgs_theta_objective(
                                    models, n_state, self.system, params_resolved,
                                    inv_vars, s1, DTYPE,
                                ),
                                x0=lbfgs_x0_theta, jac=True, method="L-BFGS-B",
                                options=lp_options_theta,
                            )
                        phys_l, data_l, _ = self._compute_losses(
                            models, n_state, self.system, params_resolved, inv_vars,
                            s1, has_obs,
                            x_obs_tf if has_obs else None,
                            y_obs_list if has_obs else None,
                            obs_si, DTYPE,
                        )
                        total_l = lambda_physics * float(phys_l) + lambda_data * float(data_l)
                        if loss_history:
                            loss_history[-1]         = total_l
                            physics_loss_history[-1] = float(phys_l)
                            data_loss_history[-1]    = float(data_l)

                # =============================================================
                # Store trajectory + cache W
                # =============================================================
                for j in range(n_state):
                    a0, a_n, b_n = models[j].coeffs()
                    y_seg = models[j].g2_from_coeffs(a0, a_n, b_n).numpy()
                    trajectory[i0:i1, j] = y_seg.astype(DTYPE.as_numpy_dtype)
                    segment_W_cache[k][j] = tf.identity(models[j].W)
                    prev_W[j] = tf.identity(models[j].W)

                # Continuity: propagate terminal state as IC for next segment
                next_init2 = []
                next_init1 = []
                for j in range(n_state):
                    a0, a_n, b_n = models[j].coeffs()
                    y_seg_tf  = models[j].g2_from_coeffs(a0, a_n, b_n)
                    dy_dx_tf  = models[j].g1_from_coeffs(a0, a_n, b_n)
                    next_init2.append(tf.cast(y_seg_tf[-1], DTYPE))
                    next_init1.append(tf.cast(dy_dx_tf[-1], DTYPE))

                init1 = next_init1
                init2 = next_init2

                if verbose:
                    last_total = loss_history[-1]         if loss_history else 0.0
                    last_phys  = physics_loss_history[-1] if physics_loss_history else 0.0
                    last_data  = data_loss_history[-1]    if data_loss_history else 0.0
                    if _is_warm:
                        pass_tag = f"warm {pass_idx-n_passes+1}/{warm_seg_passes-1}  "
                    elif _total_passes > 1:
                        pass_tag = f"pass {pass_idx+1}/{n_passes}  "
                    else:
                        pass_tag = ""
                    param_str  = "  ".join(
                        f"{nm}={ip.numpy_value:.5g}"
                        if isinstance(ip, InverseParameter)
                        else f"{nm}={float(ip):.5g}"
                        for nm, ip in inverse_params.items()
                    )
                    print(
                        f"[{pass_tag}seg {k:3d}/{n_seg-1}] "
                        f"loss={last_total:.3e}  "
                        f"phys={last_phys:.3e}  "
                        f"data={last_data:.3e}  |  {param_str}"
                    )

        t_all1 = time.perf_counter()
        if verbose:
            print(f"[InverseSolver] total elapsed: {t_all1 - t_all0:.2f} s")

        return {
            "trajectory":            trajectory,
            "t":                     t_full,
            "loss_history":          loss_history,
            "physics_loss_history":  physics_loss_history,
            "data_loss_history":     data_loss_history,
            "param_history":         param_history,
            "runtime_sec":           float(t_all1 - t_all0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_var_list(models, s1):
        """Compute (y, y_t) for each state from current ADA weights."""
        var_list = []
        for m in models:
            a0, a_n, b_n = m.coeffs()
            y     = m.g2_from_coeffs(a0, a_n, b_n)
            dy_dx = m.g1_from_coeffs(a0, a_n, b_n)
            y_t   = dy_dx * s1
            var_list.append((y, y_t))
        return var_list

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_losses(
        models, n_state, system, params_resolved, inv_vars,
        s1, has_obs, x_obs_tf, y_obs_list, obs_si, DTYPE,
        data_scales=None, phys_scales=None,
    ):
        """Compute (physics_loss, data_loss, None) as TF scalars, no tape."""
        p_list   = _build_p_list(params_resolved)
        var_list = InverseSolver._predict_var_list(models, s1)

        phys_parts = []
        for j in range(n_state):
            res = system.rhs_tf(var_list, j, p=p_list)
            if phys_scales is not None:
                res = res / (phys_scales[j] + 1e-8)
            phys_parts.append(tf.reduce_mean(tf.square(res)))
        physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

        data_loss = tf.constant(0.0, dtype=DTYPE)
        if has_obs and x_obs_tf is not None:
            d_parts = []
            for col, j_state in enumerate(obs_si):
                a0, a_n, b_n = models[j_state].coeffs()
                y_pred   = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                residual = y_pred - y_obs_list[col]
                if data_scales is not None:
                    residual = residual / (data_scales[col] + 1e-8)
                d_parts.append(tf.reduce_mean(tf.square(residual)))
            if d_parts:
                data_loss = tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)

        return physics_loss, data_loss, None

    # ------------------------------------------------------------------
    # Phase A: data pre-fit step (W only, L_data)
    # ------------------------------------------------------------------

    @staticmethod
    def _W_prefit_step(
        models, W_vars, optimizer,
        obs_si, x_obs_tf, y_obs_list, DTYPE,
        data_scales=None,
    ):
        """One W-only Adam step minimizing L_data only (Phase A)."""
        with tf.GradientTape() as tape:
            d_parts = []
            for col, j_state in enumerate(obs_si):
                a0, a_n, b_n = models[j_state].coeffs()
                y_pred   = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                residual = y_pred - y_obs_list[col]
                if data_scales is not None:
                    residual = residual / (data_scales[col] + 1e-8)
                d_parts.append(tf.reduce_mean(tf.square(residual)))
            data_loss = (
                tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)
                if d_parts else tf.constant(0.0, dtype=DTYPE)
            )

        grads = tape.gradient(data_loss, W_vars)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, W_vars)]
        optimizer.apply_gradients(zip(grads, W_vars))
        return data_loss

    # ------------------------------------------------------------------
    # Phase B: joint Adam step (W + θ, L_total)
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_step(
        models, all_trainable, n_state, system, params_resolved,
        optimizer, s1, has_obs, x_obs_tf, y_obs_list, obs_si,
        lambda_physics, lambda_data, DTYPE,
        data_scales=None, phys_scales=None,
    ):
        """One joint Adam step over W + θ  (L_total = λ_p·L_p + λ_d·L_d)."""
        with tf.GradientTape() as tape:
            p_list   = _build_p_list(params_resolved)
            var_list = InverseSolver._predict_var_list(models, s1)

            phys_parts = []
            for j in range(n_state):
                res = system.rhs_tf(var_list, j, p=p_list)
                if phys_scales is not None:
                    res = res / (phys_scales[j] + 1e-8)
                phys_parts.append(tf.reduce_mean(tf.square(res)))
            physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

            data_loss = tf.constant(0.0, dtype=DTYPE)
            if has_obs and x_obs_tf is not None:
                d_parts = []
                for col, j_state in enumerate(obs_si):
                    a0, a_n, b_n = models[j_state].coeffs()
                    y_pred   = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                    residual = y_pred - y_obs_list[col]
                    if data_scales is not None:
                        residual = residual / (data_scales[col] + 1e-8)
                    d_parts.append(tf.reduce_mean(tf.square(residual)))
                if d_parts:
                    data_loss = tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)

            total_loss = (
                tf.cast(lambda_physics, DTYPE) * physics_loss
                + tf.cast(lambda_data,   DTYPE) * data_loss
            )

        grads = tape.gradient(total_loss, all_trainable)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, all_trainable)
        ]
        optimizer.apply_gradients(zip(grads, all_trainable))
        return physics_loss, data_loss, total_loss

    # ------------------------------------------------------------------
    # JIT-compiled step factories (one compiled fn per segment)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_compiled_joint_step(
        models, all_trainable, n_state, system, params_resolved,
        optimizer, s1, has_obs, x_obs_tf, y_obs_list, obs_si,
        lambda_physics, lambda_data, DTYPE,
        data_scales=None, phys_scales=None,
    ):
        """Return a @tf.function-compiled joint Adam step for one segment.

        Called once per segment; the returned callable is then invoked
        ``epochs * adam_inner`` times with zero Python overhead per call.
        """
        lp = tf.constant(lambda_physics, dtype=DTYPE)
        ld = tf.constant(lambda_data,    dtype=DTYPE)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                p_list   = _build_p_list(params_resolved)
                var_list = InverseSolver._predict_var_list(models, s1)

                phys_parts = []
                for j in range(n_state):
                    res = system.rhs_tf(var_list, j, p=p_list)
                    if phys_scales is not None:
                        res = res / (phys_scales[j] + 1e-8)
                    phys_parts.append(tf.reduce_mean(tf.square(res)))
                physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

                data_loss = tf.constant(0.0, dtype=DTYPE)
                if has_obs and x_obs_tf is not None:
                    d_parts = []
                    for col, j_state in enumerate(obs_si):
                        a0, a_n, b_n = models[j_state].coeffs()
                        y_pred   = _eval_g2_at_x(
                            models[j_state], a0, a_n, b_n, x_obs_tf)
                        residual = y_pred - y_obs_list[col]
                        if data_scales is not None:
                            residual = residual / (data_scales[col] + 1e-8)
                        d_parts.append(tf.reduce_mean(tf.square(residual)))
                    if d_parts:
                        data_loss = tf.add_n(d_parts) / tf.cast(
                            len(d_parts), DTYPE)

                total_loss = lp * physics_loss + ld * data_loss

            grads = tape.gradient(total_loss, all_trainable)
            grads = [
                tf.zeros_like(v) if g is None else g
                for g, v in zip(grads, all_trainable)
            ]
            optimizer.apply_gradients(zip(grads, all_trainable))
            return physics_loss, data_loss, total_loss

        return step

    @staticmethod
    def _make_compiled_prefit_step(
        models, W_vars, optimizer,
        obs_si, x_obs_tf, y_obs_list, DTYPE,
        data_scales=None,
    ):
        """Return a @tf.function-compiled W-only prefit step (Phase A)."""

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                d_parts = []
                for col, j_state in enumerate(obs_si):
                    a0, a_n, b_n = models[j_state].coeffs()
                    y_pred   = _eval_g2_at_x(
                        models[j_state], a0, a_n, b_n, x_obs_tf)
                    residual = y_pred - y_obs_list[col]
                    if data_scales is not None:
                        residual = residual / (data_scales[col] + 1e-8)
                    d_parts.append(tf.reduce_mean(tf.square(residual)))
                data_loss = (
                    tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)
                    if d_parts else tf.constant(0.0, dtype=DTYPE)
                )
            grads = tape.gradient(data_loss, W_vars)
            grads = [
                tf.zeros_like(v) if g is None else g
                for g, v in zip(grads, W_vars)
            ]
            optimizer.apply_gradients(zip(grads, W_vars))
            return data_loss

        return step

    # ------------------------------------------------------------------
    # Alternating-mode steps (kept for training_strategy="alternating")
    # ------------------------------------------------------------------

    @staticmethod
    def _W_step(
        models, W_vars, optimizer,
        n_state, system, params_resolved, inv_vars,
        s1, has_obs, x_obs_tf, y_obs_list, obs_si,
        lambda_physics, lambda_data, DTYPE
    ):
        """One Adam step over W only (phys + data loss, θ frozen)."""
        with tf.GradientTape() as tape:
            p_list   = _build_p_list(params_resolved)
            var_list = InverseSolver._predict_var_list(models, s1)

            phys_parts = []
            for j in range(n_state):
                res = system.rhs_tf(var_list, j, p=p_list)
                phys_parts.append(tf.reduce_mean(tf.square(res)))
            physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

            data_loss = tf.constant(0.0, dtype=DTYPE)
            if has_obs and x_obs_tf is not None:
                d_parts = []
                for col, j_state in enumerate(obs_si):
                    a0, a_n, b_n = models[j_state].coeffs()
                    y_pred = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                    d_parts.append(tf.reduce_mean(tf.square(y_pred - y_obs_list[col])))
                if d_parts:
                    data_loss = tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)

            total_loss = (
                tf.cast(lambda_physics, DTYPE) * physics_loss
                + tf.cast(lambda_data,   DTYPE) * data_loss
            )

        grads = tape.gradient(total_loss, W_vars)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, W_vars)]
        optimizer.apply_gradients(zip(grads, W_vars))
        return physics_loss, data_loss, total_loss

    # ------------------------------------------------------------------

    @staticmethod
    def _theta_step(
        models, n_state, system, params_resolved, inv_vars,
        optimizer, s1, DTYPE
    ):
        """One Adam step over θ only (phys loss only, W frozen)."""
        with tf.GradientTape() as tape:
            p_list   = _build_p_list(params_resolved)
            var_list = InverseSolver._predict_var_list(models, s1)

            phys_parts = []
            for j in range(n_state):
                res = system.rhs_tf(var_list, j, p=p_list)
                phys_parts.append(tf.reduce_mean(tf.square(res)))
            physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

        grads = tape.gradient(physics_loss, inv_vars)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, inv_vars)]
        optimizer.apply_gradients(zip(grads, inv_vars))
        return physics_loss

    # ------------------------------------------------------------------
    # L-BFGS helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lbfgs_pack(all_trainable) -> np.ndarray:
        return np.concatenate(
            [v.numpy().reshape(-1) for v in all_trainable]
        ).astype(np.float64)

    @staticmethod
    def _lbfgs_unpack(all_trainable, w_flat: np.ndarray, DTYPE):
        offset = 0
        w_tf = tf.constant(w_flat, dtype=DTYPE)
        for v in all_trainable:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(w_tf[offset: offset + n], v.shape))
            offset += n

    @staticmethod
    def _make_lbfgs_objective(
        models, all_trainable, n_state, system, params_resolved, inv_vars,
        s1, has_obs, x_obs_tf, y_obs_list, obs_si,
        lambda_physics, lambda_data, DTYPE,
        data_scales=None, phys_scales=None,
    ):
        """Return (w_flat) -> (loss, grad) for joint L-BFGS over W + θ."""

        def objective(w_flat):
            InverseSolver._lbfgs_unpack(all_trainable, w_flat, DTYPE)

            with tf.GradientTape() as tape:
                p_list   = _build_p_list(params_resolved)
                var_list = InverseSolver._predict_var_list(models, s1)

                phys_parts = []
                for j in range(n_state):
                    res = system.rhs_tf(var_list, j, p=p_list)
                    if phys_scales is not None:
                        res = res / (phys_scales[j] + 1e-8)
                    phys_parts.append(tf.reduce_mean(tf.square(res)))
                physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

                data_loss = tf.constant(0.0, dtype=DTYPE)
                if has_obs and x_obs_tf is not None:
                    d_parts = []
                    for col, j_state in enumerate(obs_si):
                        a0, a_n, b_n = models[j_state].coeffs()
                        y_pred   = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                        residual = y_pred - y_obs_list[col]
                        if data_scales is not None:
                            residual = residual / (data_scales[col] + 1e-8)
                        d_parts.append(tf.reduce_mean(tf.square(residual)))
                    if d_parts:
                        data_loss = tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)

                total = (
                    tf.cast(lambda_physics, DTYPE) * physics_loss
                    + tf.cast(lambda_data,   DTYPE) * data_loss
                )

            grads = tape.gradient(total, all_trainable)
            grads = [
                tf.zeros_like(v) if g is None else g
                for g, v in zip(grads, all_trainable)
            ]
            loss_val = float(total.numpy())
            grad_np  = np.concatenate(
                [g.numpy().reshape(-1) for g in grads]
            ).astype(np.float64)
            return loss_val, grad_np

        return objective

    # ------------------------------------------------------------------

    @staticmethod
    def _make_lbfgs_W_objective(
        models, W_vars, n_state, system, params_resolved,
        s1, has_obs, x_obs_tf, y_obs_list, obs_si,
        lambda_physics, lambda_data, DTYPE
    ):
        """L-BFGS objective over W only (phys + data, θ frozen). [alternating mode]"""

        def objective(w_flat):
            InverseSolver._lbfgs_unpack(W_vars, w_flat, DTYPE)

            with tf.GradientTape() as tape:
                p_list   = _build_p_list(params_resolved)
                var_list = InverseSolver._predict_var_list(models, s1)

                phys_parts = []
                for j in range(n_state):
                    res = system.rhs_tf(var_list, j, p=p_list)
                    phys_parts.append(tf.reduce_mean(tf.square(res)))
                physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

                data_loss = tf.constant(0.0, dtype=DTYPE)
                if has_obs and x_obs_tf is not None:
                    d_parts = []
                    for col, j_state in enumerate(obs_si):
                        a0, a_n, b_n = models[j_state].coeffs()
                        y_pred = _eval_g2_at_x(models[j_state], a0, a_n, b_n, x_obs_tf)
                        d_parts.append(tf.reduce_mean(tf.square(y_pred - y_obs_list[col])))
                    if d_parts:
                        data_loss = tf.add_n(d_parts) / tf.cast(len(d_parts), DTYPE)

                total = (
                    tf.cast(lambda_physics, DTYPE) * physics_loss
                    + tf.cast(lambda_data,   DTYPE) * data_loss
                )

            grads = tape.gradient(total, W_vars)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, W_vars)]
            loss_val = float(total.numpy())
            grad_np  = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)
            return loss_val, grad_np

        return objective

    # ------------------------------------------------------------------

    @staticmethod
    def _make_lbfgs_theta_objective(
        models, n_state, system, params_resolved, inv_vars, s1, DTYPE
    ):
        """L-BFGS objective over θ only (phys loss only, W frozen). [alternating mode]"""

        def objective(theta_flat):
            InverseSolver._lbfgs_unpack(inv_vars, theta_flat, DTYPE)

            with tf.GradientTape() as tape:
                p_list   = _build_p_list(params_resolved)
                var_list = InverseSolver._predict_var_list(models, s1)

                phys_parts = []
                for j in range(n_state):
                    res = system.rhs_tf(var_list, j, p=p_list)
                    phys_parts.append(tf.reduce_mean(tf.square(res)))
                physics_loss = tf.add_n(phys_parts) / tf.cast(n_state, DTYPE)

            grads = tape.gradient(physics_loss, inv_vars)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, inv_vars)]
            loss_val = float(physics_loss.numpy())
            grad_np  = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)
            return loss_val, grad_np

        return objective
