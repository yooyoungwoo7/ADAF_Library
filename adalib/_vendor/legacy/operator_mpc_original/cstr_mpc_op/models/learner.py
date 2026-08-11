#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/learner.py
============================================================
Generic OperatorLearner — wraps a Problem, an LPA basis, and a vanilla
OperatorNet shared across all problems.

Loss = mean( (ẋ_pred − f(x_pred, θ))² / RES_SCALE² )
       + cons_w · mean( ((H − H₀) / |H₀|)² )       [if problem provides H]

Phase-1 fit() — single-segment training.
fit_rollout() is retained as an opt-in diagnostic; default ROLLOUT_EPOCHS=0
in config disables it. Inference uses rollout_full_trajectory() which chains
N_SEG segments with NO gradient.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from config import (
    TF_DTYPE, INPUT_DIM, RES_SCALE, GRAD_CLIP_NORM, USE_JIT, BASIS_NAME,
)
from models.basis import build_basis
from models.operator_net import OperatorNet
from utils.io_utils import ensure_dir

DTYPE = getattr(tf, TF_DTYPE)


# ============================================================
# LR schedule
# ============================================================
class CosineAnnealingWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.peak_lr = tf.constant(float(peak_lr), dtype=tf.float32)
        self.min_lr = tf.constant(float(min_lr), dtype=tf.float32)
        self.total_steps = tf.constant(float(total_steps), dtype=tf.float32)
        self.warmup_steps = tf.constant(float(warmup_steps), dtype=tf.float32)
        self._pi = tf.constant(np.pi, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(self._pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": float(self.peak_lr.numpy()),
            "total_steps": float(self.total_steps.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "min_lr": float(self.min_lr.numpy()),
        }


# ============================================================
# Learner
# ============================================================
class OperatorLearner:
    def __init__(self, problem, hidden=256, n_layers=3, lr=1e-3,
                 x_mean=None, x_std=None, basis_name=None):
        self.problem = problem
        self.basis = build_basis(basis_name or BASIS_NAME)
        self.basis_name = (basis_name or BASIS_NAME).lower()
        # If the problem provides physics-informed derived features, wire them
        # into the OperatorNet. Otherwise behaves as a vanilla MLP.
        derived_fn = getattr(problem, "derived_features_tf", None)
        n_derived  = int(getattr(problem, "n_derived_features", 0) or 0)
        if n_derived == 0:
            derived_fn = None
        self.net = OperatorNet(
            input_dim=problem.input_dim, state_dim=problem.state_dim,
            N_p=self.basis.coef_dim,
            hidden=hidden, n_layers=n_layers,
            x_mean=x_mean, x_std=x_std,
            derived_fn=derived_fn,
            n_derived=n_derived,
            derived_mean=getattr(problem, "derived_mean", None),
            derived_std=getattr(problem,  "derived_std",  None),
            output_scale=getattr(problem, "output_scale", None),
        )
        self.base_lr = float(lr)
        self.optimizer = tf.keras.optimizers.Adam(self.base_lr)
        self.grad_clip_norm = float(GRAD_CLIP_NORM)
        self.history = {k: [] for k in ["phys_loss", "val_phys_loss", "lr"]}

        self._res_scale_tf = tf.constant(np.asarray(RES_SCALE, dtype=np.float32), dtype=DTYPE)
        self._cons_w_tf = tf.constant(float(problem.cons_w), dtype=DTYPE)
        self._log_floor = tf.constant(1e-6, dtype=DTYPE)

    # --- core forward / loss ---
    def _xdot0(self, x0, theta):
        """ẋ(0) = f(x0, theta) — needed by bases that hard-enforce ẋ(0)."""
        return self.problem.rhs_tf(x0, theta)

    def forward_segment(self, X_batch, training=False):
        X_batch = tf.convert_to_tensor(X_batch, dtype=DTYPE)
        x0, theta = self.problem.split_input_tf(X_batch, dtype=DTYPE)
        W_pred = self.net(X_batch, training=training)
        xdot0 = self._xdot0(x0, theta)
        return self.basis(W_pred, x0, xdot0)

    def _residual_loss(self, x, xdot, theta):
        res = (tf.convert_to_tensor(xdot, dtype=DTYPE) - self.problem.rhs_tf(x, theta)) / self._res_scale_tf
        return tf.reduce_mean(tf.square(res))

    def _conservation_loss(self, x, theta):
        H = self.problem.conservation_quantity(x, theta)
        if H is None:
            return tf.constant(0.0, dtype=DTYPE)
        H0 = H[:, 0:1]
        rel = (H - H0) / (tf.abs(H0) + self._log_floor)
        return tf.reduce_mean(tf.square(rel))

    def _physics_loss(self, x, xdot, theta):
        res = self._residual_loss(x, xdot, theta)
        if self.problem.cons_w > 0:
            return res + self._cons_w_tf * self._conservation_loss(x, theta)
        return res

    def compute_physics_loss(self, X_batch, training=False):
        out = self.forward_segment(X_batch, training=training)
        _, theta = self.problem.split_input_tf(X_batch, dtype=DTYPE)
        loss = self._physics_loss(out["x"], out["xdot"], theta)
        out["phys_loss"] = loss
        return out

    @tf.function(reduce_retracing=True, jit_compile=USE_JIT)
    def train_step(self, X_batch):
        with tf.GradientTape() as tape:
            out = self.compute_physics_loss(X_batch, training=True)
            loss = out["phys_loss"]
        grads = tape.gradient(loss, self.net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True, jit_compile=USE_JIT)
    def eval_step(self, X_batch):
        return self.compute_physics_loss(X_batch, training=False)["phys_loss"]

    # --- rollout (DIAGNOSTIC ONLY) ---
    def _rollout_physics(self, X_batch, K, alpha, training):
        x0, theta = self.problem.split_input_tf(X_batch, dtype=DTYPE)
        xk = x0
        phys_sum = tf.constant(0.0, dtype=DTYPE)
        phys_0 = tf.constant(0.0, dtype=DTYPE)
        for k in range(K):
            Xk = tf.concat([xk, theta], axis=-1)
            W = self.net(Xk, training=training)
            xdotk = self._xdot0(xk, theta)
            out = self.basis(W, xk, xdotk)
            phys_k = self._physics_loss(out["x"], out["xdot"], theta)
            if k == 0:
                phys_0 = phys_k
            phys_sum = phys_sum + phys_k
            xk = out["x_end"]
        rollout_mean = phys_sum / float(K)
        loss = alpha * rollout_mean + (1.0 - alpha) * phys_0
        return loss, rollout_mean, phys_0

    def _make_rollout_steps(self, K, alpha):
        K = int(K)
        alpha = tf.constant(float(alpha), dtype=DTYPE)

        @tf.function(reduce_retracing=True, jit_compile=USE_JIT)
        def train_step_rollout(X_batch):
            with tf.GradientTape() as tape:
                loss, rmean, p0 = self._rollout_physics(X_batch, K, alpha, training=True)
            grads = tape.gradient(loss, self.net.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
            return loss, rmean, p0

        @tf.function(reduce_retracing=True, jit_compile=USE_JIT)
        def eval_step_rollout(X_batch):
            return self._rollout_physics(X_batch, K, alpha, training=False)

        return train_step_rollout, eval_step_rollout

    def fit_rollout(self, train_ds, val_ds=None, epochs=500, K=3, alpha=0.75,
                    fixed_lr=None, checkpoint_dir=None, print_every=10):
        if fixed_lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(float(fixed_lr))
            print(f"[rollout-DIAGNOSTIC] fixed lr={float(fixed_lr):.2e}, K={K}, alpha={alpha}")
        for k in ("rollout_loss", "rollout_mean", "rollout_phys_0", "val_rollout_loss", "val_rollout_mean"):
            self.history.setdefault(k, [])
        train_step, eval_step = self._make_rollout_steps(K, alpha)

        best_val = np.inf
        best_path = None
        bar = tqdm(range(1, epochs + 1), desc=f"rollout K={K}", unit="ep")
        for ep in bar:
            t_ep = time.perf_counter()
            tr_loss, tr_mean, tr_p0 = [], [], []
            for X_batch, _ in train_ds:
                loss, rmean, p0 = train_step(X_batch)
                tr_loss.append(float(loss.numpy()))
                tr_mean.append(float(rmean.numpy()))
                tr_p0.append(float(p0.numpy()))
            self.history["rollout_loss"].append(float(np.mean(tr_loss)))
            self.history["rollout_mean"].append(float(np.mean(tr_mean)))
            self.history["rollout_phys_0"].append(float(np.mean(tr_p0)))

            val_mean = np.nan
            val_r = np.nan
            if val_ds is not None:
                v_loss, v_mean = [], []
                for X_batch, _ in val_ds:
                    loss, rmean, _ = eval_step(X_batch)
                    v_loss.append(float(loss.numpy()))
                    v_mean.append(float(rmean.numpy()))
                val_mean = float(np.mean(v_loss))
                val_r = float(np.mean(v_mean))
                self.history["val_rollout_loss"].append(val_mean)
                self.history["val_rollout_mean"].append(val_r)
                if checkpoint_dir is not None and val_mean < best_val:
                    best_val = val_mean
                    best_path = self.save_weights(checkpoint_dir, ep, tag="best_rollout")

            ep_sec = time.perf_counter() - t_ep
            bar.set_postfix(loss=f"{np.mean(tr_loss):.3e}",
                            p0=f"{np.mean(tr_p0):.3e}",
                            val=f"{val_mean:.3e}" if val_ds is not None else "n/a",
                            ep_s=f"{ep_sec:.2f}")
            if ep == 1 or ep % print_every == 0 or ep == epochs:
                msg = (f"[rollout {ep:4d}/{epochs}] loss={np.mean(tr_loss):.6e} "
                       f"(mean={np.mean(tr_mean):.3e}, p0={np.mean(tr_p0):.3e})")
                if val_ds is not None:
                    msg += f" | val_loss={val_mean:.6e} (mean={val_r:.3e})"
                msg += f" | {ep_sec:.2f}s"
                tqdm.write(msg)

        return {
            "best_rollout_checkpoint": best_path,
            "best_val_rollout_loss": None if not np.isfinite(best_val) else best_val,
        }

    # --- physics-only training loop ---
    def _build_schedule(self, steps_per_epoch, epochs, warmup_fraction=0.05, min_lr=1e-6):
        total = steps_per_epoch * epochs
        warmup = int(total * warmup_fraction)
        sched = CosineAnnealingWarmup(self.base_lr, total, warmup, min_lr=min_lr)
        self.optimizer = tf.keras.optimizers.Adam(sched)
        return sched

    def fit(self, train_ds, val_ds=None, epochs=100, checkpoint_dir=None, print_every=10,
            use_lr_schedule=True, warmup_fraction=0.05, min_lr=1e-6):
        steps_per_epoch = sum(1 for _ in train_ds)
        sched = None
        if use_lr_schedule:
            sched = self._build_schedule(steps_per_epoch, epochs, warmup_fraction, min_lr)
            print(f"[LR] cosine warmup: peak={self.base_lr:.1e}, "
                  f"warmup={int(steps_per_epoch*epochs*warmup_fraction)}, "
                  f"total={steps_per_epoch*epochs}, min={min_lr:.1e}")

        best_val = np.inf
        best_path = None
        bar = tqdm(range(1, epochs + 1), desc="physics", unit="ep")
        for ep in bar:
            t_ep = time.perf_counter()
            tr = []
            for X_batch, _ in train_ds:
                loss = self.train_step(X_batch)
                tr.append(float(loss.numpy()))
            tr_mean = float(np.mean(tr))
            self.history["phys_loss"].append(tr_mean)
            current_lr = float(sched(self.optimizer.iterations)) if sched is not None else float(self.base_lr)
            self.history["lr"].append(current_lr)

            val_mean = np.nan
            if val_ds is not None:
                vals = []
                for X_batch, _ in val_ds:
                    vals.append(float(self.eval_step(X_batch).numpy()))
                val_mean = float(np.mean(vals))
                self.history["val_phys_loss"].append(val_mean)
                if checkpoint_dir is not None and val_mean < best_val:
                    best_val = val_mean
                    best_path = self.save_weights(checkpoint_dir, ep, tag="best")

            ep_sec = time.perf_counter() - t_ep
            bar.set_postfix(phys=f"{tr_mean:.3e}",
                            val=f"{val_mean:.3e}" if val_ds is not None else "n/a",
                            lr=f"{current_lr:.2e}",
                            ep_s=f"{ep_sec:.2f}")
            if ep == 1 or ep % print_every == 0 or ep == epochs:
                lr_str = f" lr={current_lr:.2e}" if sched else ""
                tqdm.write(f"[{ep:4d}/{epochs}] phys={tr_mean:.6e}"
                           + (f" | val={val_mean:.6e}" if val_ds is not None else "")
                           + lr_str + f" | {ep_sec:.2f}s")

        return {"best_checkpoint": best_path,
                "best_val_phys_loss": None if not np.isfinite(best_val) else best_val}

    # --- checkpoints ---
    def save_weights(self, ckpt_dir, epoch, tag=None):
        ensure_dir(ckpt_dir)
        dummy = tf.zeros((1, INPUT_DIM), dtype=DTYPE)
        _ = self.net(dummy, training=False)
        name = f"epoch_{epoch:05d}" + (f"_{tag}" if tag else "")
        path = f"{ckpt_dir}/{name}.weights.h5"
        self.net.save_weights(path)
        return path

    def load_weights(self, path):
        dummy = tf.zeros((1, INPUT_DIM), dtype=DTYPE)
        _ = self.net(dummy, training=False)
        self.net.load_weights(path)
        print(f"[LOAD] weights <- {path}")

    # --- inference ---
    def predict_segment(self, X_np):
        X = tf.convert_to_tensor(np.asarray(X_np, dtype=np.float32), dtype=DTYPE)
        if len(X.shape) == 1:
            X = X[None, :]
        out = self.forward_segment(X, training=False)
        return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in out.items()}

    def rollout_full_trajectory(self, x_input, n_seg):
        """Inference-only rollout: chain n_seg segments by feeding x_end → x_0.

        Per project guidance, this is NOT used during training — the operator
        learns one segment well across diverse ICs and the rollout works at
        inference because x(0) = x_0 is structurally enforced inside the basis.
        """
        problem = self.problem
        x_input = np.asarray(x_input, dtype=np.float32).reshape(-1)
        theta = x_input[problem.state_dim:problem.state_dim + problem.param_dim].copy()
        xk = x_input[:problem.state_dim].copy()

        t_segs, x_segs, res_segs = [], [], []
        t_offset = 0.0
        for k in range(int(n_seg)):
            Xk = np.concatenate([xk, theta], axis=0)[None, :].astype(np.float32)
            out = self.predict_segment(Xk)
            x_seg = out["x"][0]
            xdot_seg = out["xdot"][0]
            res = (tf.convert_to_tensor(xdot_seg[None, ...], dtype=DTYPE)
                   - problem.rhs_tf(tf.constant(x_seg[None, ...], dtype=DTYPE),
                                    tf.constant(theta[None, :], dtype=DTYPE))).numpy()[0]
            t_local = self.basis.t_local_np + t_offset
            if k > 0:
                t_local = t_local[1:]
                x_seg = x_seg[1:]
                res = res[1:]
            t_segs.append(t_local)
            x_segs.append(x_seg)
            res_segs.append(res)
            t_offset += self.basis.T_seg
            xk = out["x_end"][0]

        return {
            "t": np.concatenate(t_segs, axis=0),
            "x": np.concatenate(x_segs, axis=0),
            "residual": np.concatenate(res_segs, axis=0),
            "theta": theta,
        }
