# Project: Operator Library

Unified physics-informed neural-operator library with selectable basis (LPA
or ADA-F) and problem (4 bundled). Switch via `config.PROBLEM_NAME` /
`BASIS_NAME`, env vars `PROBLEM=` / `BASIS=`, or CLI `--problem` /
`--basis`. See `WORKFLOW.md` / `WORKFLOW_EN.md` for the full pipeline.

## Quick reference

```
config.PROBLEM_NAME          → problems/registry.get_problem(...)
config.BASIS_NAME            → models/basis.get_basis_cls(...)
PROBLEM_CONFIGS[name]        → resolved into module-level T_FINAL, N_SEG, …
OperatorLearner(problem,     → builds OperatorNet (zero-init, optional
  basis_name=...)              output_scale) + chosen basis
fit()                        → single-segment physics training (canonical)
fit_rollout()                → opt-in K-segment diagnostic; default disabled
rollout_full_trajectory()    → INFERENCE-ONLY chained rollout (no gradient)
```

## Bases

| name    | hard IC                          | output dim     | constants in config |
|---------|----------------------------------|---------------|---------------------|
| `lpa`   | `x(0) = x0`                      | `LPA_N_P`     | `LPA_N_P`, `MAX_ORDER` per problem |
| `adaf`  | `x(0) = x0` AND `ẋ(0) = f(x0,θ)` | `ADAF_N_P`    | `ADAF_N_M`, `ADAF_N_P` (library-wide) |

Both share `basis(W, x0, xdot0)` signature and output dict; the learner
passes `xdot0 = problem.rhs_tf(x0, θ)` regardless of basis (LPA ignores it).

## Adding a new problem
1. `problems/<name>_problem.py` — subclass `BaseProblem`, implement
   `sample_cases / rhs_np / rhs_tf`. Optional hooks (any subset):
   - `conservation_quantity(x, theta)` + `cons_w > 0` — Hamiltonian penalty.
   - `derived_features_tf(x_input)` + `n_derived_features` + `derived_mean/std` —
     physics-informed input augmentation (LV V₀, CSTR k1/k2/k3, Haldane μ, …).
   - `output_scale = RES_SCALE` — recommended whenever RES_SCALE is
     non-uniform across states (balances `∂Loss/∂W`). Leave None for uniform
     RES_SCALE problems; Adam absorbs constant scalings.
   - `diverse_random_inputs(n, rng)` — hand-picked dynamical archetypes for
     `plot_random_cases`.
   - `extra_plot_traces()` — auxiliary rows below state rows in plots
     (CSTR uses this for ΔT, F, Q).
   - `segment_sampling_strategy = "all"` — build every segment per case
     instead of one random pick (CSTR uses this so the brief Arrhenius
     transient is always present).
   - `apply_train_oversampling(seg)` — optional row-level row duplication
     at train load time. Default identity. (Currently unused by all 4
     bundled problems; was previously used by CSTR but removed as the
     wall-clock cost outweighed accuracy gain.)
2. Register in `problems/registry.py`.
3. Add entry to `config.PROBLEM_CONFIGS`.
No other file needs editing.

## Per-problem notable tuning
- **lotka**: (U, R) 2-D family, 6 LV-physics derived features, conservation
  loss (`cons_w = 0.05`), `MAX_ORDER = 10`, `T_FINAL = 1.0`. Uniform
  RES_SCALE → `output_scale = None`.
- **bioreactor**: 4-state Haldane fed-batch, 6 Haldane derived features,
  `T_FINAL = 100`, `N_SEG = 50`, large training set (50K cases × 1 random
  seg). Non-uniform RES_SCALE = `(0.033, 3.4, 0.025, 0.1)` →
  **`output_scale = RES_SCALE`** (balances 18,000× gradient imbalance).
- **cstr**: 4-state Arrhenius + heat balance, **stiff** → BDF solver;
  `MAX_ORDER = 10`, `HIDDEN = 128`, `RES_SCALE = (0.5, 0.3, 30, 15)` →
  **`output_scale = RES_SCALE`** (3,600× imbalance), `T_FINAL = 0.5 h`,
  `segment_sampling_strategy = "all"` (every transient seg always present).
  Param vector is `[α, β, F, Q]` (Q renamed from Q_dot for clarity); F is
  internally a dilution rate `[1/h]` but **displayed in `l/min`** via
  `F·V_R/60` in `extra_plot_traces` and `case_subtitle`. 7-row plot:
  states + ΔT + F + Q.
- **triple_tank**: 3-state Torricelli, 6 Torricelli derived features
  (Q13, Q32, Q20, per-tank net inflows), `T_FINAL = 300 s`. Uniform
  RES_SCALE → `output_scale = None`.

## Physics-informed derived features
A problem may inject extra physics-aware columns into the network input by
overriding `derived_features_tf(x_input)` and setting `n_derived_features`,
`derived_mean`, `derived_std`. The shared vanilla `OperatorNet` auto-detects
these and concatenates `raw input + derived features` before the MLP.
Each of the bundled 4 problems registers 6 derived features (LV Hamiltonian,
Haldane μ, Arrhenius rates, Torricelli flows, etc.). A problem that does not
override remains a plain MLP on the raw input.

## Per-state output scaling
Problems with non-uniform `RES_SCALE` should set `output_scale = RES_SCALE`
on their `BaseProblem` subclass. The OperatorNet multiplies the raw W output
by `output_scale[state, None]` per state — since both bases are linear in W,
this scales `x_delta`, `ẋ`, `ẍ` for state i uniformly without breaking the
hard IC. Effect: `∂Loss/∂raw` becomes roughly state-uniform so Adam doesn't
waste epochs calibrating per-state magnitudes.

## Zero-init last layer
The OperatorNet's projection to W is zero-initialised (kernel + bias). At
the first forward, W=0, so:
- LPA → `x(t) = x0` (identity)
- ADA-F → `x(t) = x0 + ẋ0·t` (affine free-part only)

This is essential for stiff problems: CSTR with `s₁ ≈ 100` would otherwise
amplify random initial weights into states like T_R = 380 °C, where
Arrhenius `exp()` overflows and the loss goes NaN before any gradient step
can correct it.

## Matplotlib Figure Settings

Apply these as global defaults for every new matplotlib figure unless
explicitly told otherwise:

```python
from utils._style import apply_style
apply_style()
```

### Figure style rules
- **Grid off** — do not re-enable `ax.grid(True)` on individual axes.
- **Tight x-limits** — `utils/_style.tight_x(ax, x)` calls `ax.set_xlim(x[0], x[-1])`.
- **Minor ticks always visible** — rcParams handles it globally.
- Save with `fig.savefig(path, bbox_inches="tight")` and call
  `fig.tight_layout()` before saving.
- Reference curves in solid black `lw=1.8`; operator/model predictions in
  `C3` (red) dashed `lw=1.5`; palette otherwise follows `C0, C1, C3, C4`.
- Axis labels and titles use LaTeX math; time unit comes from
  `problem.time_unit`.

## Progress Bars
- Use `tqdm` for long loops. Per-epoch loops use `unit="ep"`; per-batch inner
  loops use `unit="steps"` with `unit_scale=False`.
- Always display loss / validation / LR / per-iteration time via
  `set_postfix`.

## Error Metrics
Canonical metrics live in `utils/metrics.py`:
- `l1_rel`, `l2_rel`
- `statewise_l1_rel`, `statewise_l2_rel`
- `final_state_error`, `residual_stats`

Evaluation must report L2 relative error (total + per-state) at minimum.

## Code Conventions
- `USE_GPU=0` by default; CPU + XLA. GPU only on CUDA machines.
- All TF training steps are JIT-compiled via `@tf.function(jit_compile=USE_JIT)`.
- **Both bases enforce `x(0)=x0` symbolically.** The network predicts W-panel
  weights only; basis composition (Legendre or Fourier integration) is in
  dedicated code. Do NOT change this to have the network predict spectral
  amplitudes directly.
- **ADA-F additionally enforces `ẋ(0)=f(x0,θ)` symbolically** via a `(ξ+1)`
  shift in the free-part construction. The learner anchors the second IC by
  passing `xdot0 = problem.rhs_tf(x0, θ)` to the basis.
- **Single-segment training is canonical.** Long-horizon rollout is
  inference-only chaining; K-segment chained-gradient training is opt-in
  diagnostic only (default `ROLLOUT_EPOCHS = 0`).
- **Per-problem state physics never leaks into the library core.** All
  problem-specific constants (Arrhenius, Haldane, Torricelli, LV) live in
  `problems/<name>_problem.py` only. The learner reads only the
  `BaseProblem` interface.
- Training is label-free (physics + optional conservation residual). Reference
  trajectories are used only for validation / plotting.
