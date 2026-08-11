#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problems/base_problem.py
============================================================
Abstract Problem interface. Every concrete problem subclass implements:

    Required physics:
      • sample_cases(n, seed)  → (x0, theta) numpy arrays
      • rhs_np(t, x, theta)    → numpy ODE RHS
      • rhs_tf(x, theta)       → TF ODE RHS (broadcasting-aware)

    Optional:
      • conservation_quantity(x, theta) → (B, Nt) tensor or None
      • cons_w                          → float weight on conservation loss
      • derived_features_tf(x_input)    → (B, n_derived_features) tensor or None
      • n_derived_features              → int (must match the function output)
      • derived_mean / derived_std      → np arrays of shape (n_derived_features,)
      • segment_sampling_strategy       → "random" (1 segment/case) or "all" (N_SEG/case)
      • apply_train_oversampling(seg_dict) → modified dict (default = identity)
      • initial_state_extras(...)       → metadata to store alongside dataset

    Metadata for plotting:
      • state_labels, param_labels      tuples
      • time_unit                       string
      • nominal_input()                 typical (x0+θ) point
      • sweep_specs()                   list of (label, values, idx_in_input)

The generic learner (models/learner.py) and dataset_builder
(data/dataset_builder.py) consume only this interface.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

from config import NP_DTYPE


class BaseProblem:
    # ----- mandatory class attributes (override in subclass) -----
    name: str               = "base"
    state_dim: int          = 0
    param_dim: int          = 0
    state_labels: tuple     = ()
    param_labels: tuple     = ()
    time_unit: str          = ""

    # Plot-only multiplier applied to internal time before drawing.
    # E.g., CSTR uses internal t in hours but displays t·60 minutes.
    # Default 1.0 = display in the same unit as the simulation grid.
    time_factor: float      = 1.0

    # solver defaults — subclass may override
    ref_solver: str         = "RK45"
    ref_rtol: float         = 1e-9
    ref_atol: float         = 1e-11

    # conservation loss — subclass may override (set cons_w > 0 + override
    # conservation_quantity to enable)
    cons_w: float           = 0.0

    # derived input features (physics-informed) — subclass may override.
    # Default: no extras (vanilla OperatorNet input = raw input).
    n_derived_features: int = 0
    derived_mean = None  # np.ndarray of shape (n_derived_features,) or None → zeros
    derived_std  = None  # np.ndarray of shape (n_derived_features,) or None → ones

    # Per-state output scaling on the network's W output. Useful when state
    # magnitudes span orders of magnitude (e.g., CSTR: C ~ 1, T ~ 130). Set to
    # an array of shape (state_dim,) — typically equal to RES_SCALE so that
    # ∂Loss/∂W becomes roughly state-uniform. Default None = no scaling.
    output_scale = None

    # Segment-sampling strategy used by data/dataset_builder:
    #   "random" — one random segment per fullcase trajectory (default)
    #   "all"    — every one of the N_SEG segments per fullcase
    # CSTR overrides to "all" so transient + plateau both appear in training.
    segment_sampling_strategy: str = "random"

    # ---------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        return self.state_dim + self.param_dim

    # ---------------------------------------------------------------
    # MANDATORY OVERRIDES
    # ---------------------------------------------------------------
    def sample_cases(self, n: int, seed: int):
        """Return (x0, theta, meta_dict). x0: (n, state_dim). theta: (n, param_dim).
        meta_dict: optional dict of extra arrays to save alongside (e.g. {"U": U, "R": R})."""
        raise NotImplementedError

    def rhs_np(self, t: float, x, theta):
        """Numpy ODE RHS. x: (state_dim,), theta: (param_dim,). Returns (state_dim,)."""
        raise NotImplementedError

    def rhs_tf(self, x, theta):
        """TF ODE RHS with broadcasting:
            x: (..., state_dim) (typically (B, Nt, state_dim))
            theta: (B, param_dim)
        Returns (..., state_dim)."""
        raise NotImplementedError

    # ---------------------------------------------------------------
    # OPTIONAL OVERRIDES
    # ---------------------------------------------------------------
    def conservation_quantity(self, x, theta):
        """Return scalar field (B, Nt) or None if no conservation.
        Default: None — no conservation penalty."""
        return None

    def derived_features_tf(self, x_input):
        """Optional physics-informed input augmentation.

        x_input : (B, input_dim) raw network input  [x0; theta]
        returns : (B, n_derived_features) tensor, or None to disable.

        Default: None (= vanilla OperatorNet input). Override in subclass and
        set `n_derived_features > 0` to enable physics-aware feature injection.
        """
        return None

    def apply_train_oversampling(self, segment_dict):
        """Optional row-level oversampling applied at TRAIN-time only (val/test
        unchanged). Subclass may override to repeat segments matching some
        criterion (e.g., CSTR repeats early-transient segments).
        Default: identity."""
        return segment_dict

    # ---------------------------------------------------------------
    # GENERIC HELPERS (do not override)
    # ---------------------------------------------------------------
    def solve_reference(self, theta, x0, t_grid):
        """RK45/BDF reference trajectory using the numpy RHS."""
        theta = np.asarray(theta, dtype=np.float64)
        x0 = np.asarray(x0, dtype=np.float64)
        t_grid = np.asarray(t_grid, dtype=np.float64)
        sol = solve_ivp(
            lambda t, y: self.rhs_np(t, y, theta),
            (float(t_grid[0]), float(t_grid[-1])),
            x0,
            t_eval=t_grid,
            method=self.ref_solver,
            rtol=self.ref_rtol,
            atol=self.ref_atol,
        )
        if not sol.success:
            raise RuntimeError(f"{self.name} reference solve failed: {sol.message}")
        return sol.y.T.astype(NP_DTYPE)

    def split_input_tf(self, X, dtype=tf.float32):
        X = tf.convert_to_tensor(X, dtype=dtype)
        return X[..., :self.state_dim], X[..., self.state_dim:self.state_dim + self.param_dim]

    def build_input(self, x0, theta):
        return np.concatenate([np.asarray(x0), np.asarray(theta)], axis=-1).astype(NP_DTYPE)

    # ---------------------------------------------------------------
    # PLOTTING METADATA — sensible defaults; subclass may override
    # ---------------------------------------------------------------
    def nominal_input(self) -> np.ndarray:
        """Default (x0+θ) used as the held-fixed nominal during sweeps.
        Default: zeros — subclass should override."""
        return np.zeros(self.input_dim, dtype=NP_DTYPE)

    def sweep_specs(self) -> List[Tuple[str, np.ndarray, int]]:
        """List of (display_label, values, idx_in_input_vector) for sweep plots.
        Default: empty (no sweeps)."""
        return []

    def random_input(self, rng) -> np.ndarray:
        """One fresh random (x0+θ) sample for plot_random_cases.
        Default: delegate to sample_cases(1)."""
        x0, theta, _ = self.sample_cases(1, seed=int(rng.integers(0, 2**31 - 1)))
        return self.build_input(x0[0], theta[0])

    def diverse_random_inputs(self, n_cases, rng) -> list:
        """List of n diverse (x0+θ) samples for plot_random_cases.

        Default: independent uniform random_input() calls. Subclasses may
        override to return stratified samples or hand-picked dynamical
        archetypes that produce visibly distinct trajectories.
        """
        return [self.random_input(rng) for _ in range(int(n_cases))]

    def case_subtitle(self, x_input: np.ndarray) -> str:
        """Short multi-line text shown above each case in comparison/random plots.
        Default: comma-separated states + params."""
        x0 = x_input[:self.state_dim]
        theta = x_input[self.state_dim:self.state_dim + self.param_dim]
        st = ", ".join(f"{n}={v:.3g}" for n, v in zip(self.state_labels, x0))
        pt = ", ".join(f"{n}={v:.3g}" for n, v in zip(self.param_labels, theta))
        return f"{st}\n{pt}"

    def state_units(self) -> Tuple[str, ...]:
        """Per-state unit strings (used in plot ylabels). Default: empty."""
        return tuple("" for _ in self.state_labels)

    def state_plot_labels(self) -> Tuple[str, ...]:
        """LaTeX-formatted state labels for matplotlib (math mode rendering).
        Default: same as `state_labels`. Override to inject subscripts/Greek
        letters that should display nicely in plots while keeping plain
        `state_labels` for CSV/console output."""
        return tuple(self.state_labels)

    def extra_plot_traces(self):
        """Return a list of auxiliary traces to plot below the state rows.

        Each item is a dict:
            {
              "label": str,                     # ylabel (LaTeX OK)
              "unit":  str,                     # appended in brackets
              "fn":    callable(x, theta) -> 1D np.array,   # over the t-axis
              "kind":  "derived" | "constant",
            }
        - "derived": fn returns time-varying values; both reference and
          operator are plotted (uses x of shape (Nt, state_dim) per call).
        - "constant": fn returns a flat 1D array (e.g. an input that is
          held constant within a segment); a single horizontal line is drawn.

        Default: [] (no extras). CSTR overrides with ΔT, F, Q̇ for extended
        plots."""
        return []
