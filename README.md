# ADAlib (`adalib-IFL`)

**ADAlib** is a Python library built on the Anti-Derivative Approximator (ADA)
representation for ordinary differential equations. One library covers four
tasks: forward simulation, physics-informed inverse parameter estimation,
amortized operator learning, and model predictive control (MPC) with the
trained operator as a differentiable, batchable surrogate.

- Documentation: https://adaf-library.readthedocs.io/en/latest/
- Source: https://github.com/yooyoungwoo7/ADAF_Library

---

## Install

From PyPI:

```bash
pip install adalib-IFL
```

or the latest version from GitHub:

```bash
pip install git+https://github.com/yooyoungwoo7/ADAF_Library.git
```

or from a local clone:

```bash
git clone https://github.com/yooyoungwoo7/ADAF_Library.git
cd ADAF_Library
pip install -e .
```

Requires Python >= 3.10 and TensorFlow >= 2.13. Tested with Python 3.13.5 and
TensorFlow 2.20.0; `requirements.txt` pins the exact tested environment
(`pip install -r requirements.txt`).

```python
import adalib   # the PyPI distribution is "adalib-IFL"; the import name is "adalib"
```

> **Note:** an unrelated PyPI package called `adalib` uses the same import
> name. Do not install both in the same environment.

---

## Feature support

| Feature | User-defined `CallableODESystem` | Built-in systems |
|---|:---:|:---:|
| Forward | ✅ Fully supported | ✅ Supported |
| Operator learning | ✅ Physics-residual only (LPA Operator NN) | ✅ Supported |
| MPC — tracking | ✅ LPA Operator NN surrogate | ✅ Supported |
| MPC — economic | ❌ Not yet supported | ✅ Supported |
| Inverse (parameter estimation) | ⚠️ Works; validated on LV/Euler only | ⚠️ Validated on LV/Euler only |

**Built-in systems:** `cstr`, `triple_tank`, `fedbatch_bioreactor`,
`lotka_volterra`, `lotka_volterra_ur`, `euler`, `damped_pendulum`
(see [Built-in systems](#built-in-systems)).

> **Inverse caveat:** the inverse solver is validated only for
> `lotka_volterra` and `euler` under low noise and full observation. Inverse
> estimation for `fedbatch_bioreactor` is known to fail (the Haldane kinetics
> are poorly identifiable), and for `cstr` / `triple_tank` it is unverified.

---

## Quick start

### 1. Forward — user-defined ODE

```python
import adalib

def rhs(t, x, u=None, p=None):
    return [-x[0]]                    # dy/dt = -y

def rhs_tf(var_list, i, u=None, p=None):
    y, y_t = var_list[0]
    return y_t - (-y)                 # ADA-F physics residual

system  = adalib.CallableODESystem("decay", rhs, rhs_tf=rhs_tf, state_names=["y"])
options = adalib.ForwardOptions(basis="adaf", n_seg=10, epochs=5, use_lbfgs=True)
result  = adalib.run_forward(system=system, x0=[1.0], t_span=(0.0, 3.0), options=options)

t = result.solution.t   # (Nt_total,)
y = result.solution.y   # (n_state, Nt_total)
```

### 2. Operator learning — built-in system

```python
import adalib

system  = adalib.get_system("cstr")
options = adalib.OperatorOptions(
    basis="lpa", n_train=2000, epochs=1000,
    work_dir="./runs/cstr_operator",
)
result  = adalib.run_operator(
    system=system,
    x0=[0.8, 0.5, 134.14, 130.0],
    t_span=(0.0, 0.5),
    options=options,
)

t, y = result.t, result.y          # rollout at segment boundaries
print(result.paths["work_dir"])     # all artifacts saved here
```

### 3a. MPC — built-in system (legacy backend)

```python
import adalib

system  = adalib.get_system("cstr")
options = adalib.MPCOptions(
    mode="tracking", basis="lpa",
    target={"T_R": 136.0}, n_steps=20,
    n_train=200, epochs=300,
    work_dir="./runs/cstr_mpc",
)
result  = adalib.run_mpc(
    system=system,
    x0=[0.8, 0.5, 141.0, 141.0],
    options=options,
)

t, x, u = result.t, result.x, result.u   # closed-loop plant trajectory
```

### 3b. MPC — user-defined system (generic tracking)

```python
import adalib

def msd_rhs(t, state, u=None, p=None):
    x, v = state
    F = u[0] if u else 0.0
    m, c, k = 1.0, 0.3, 1.5
    return [v, (F - c * v - k * x) / m]

system = adalib.CallableODESystem(
    name="mass_spring_damper",
    rhs=msd_rhs,
    state_names=["x", "v"],
    control_names=["F"],
    state_bounds={"x": (-3.0, 3.0), "v": (-4.0, 4.0)},
    control_bounds={"F": (-5.0, 5.0)},
)

options = adalib.MPCOptions(
    mode="tracking",
    controlled_variables=["x"],
    target={"x": 1.0},
    dt=0.4, horizon=5,
    tracking_weights=[10.0], control_weights=[0.05],
    n_train=400, n_val=80, generate_data=True, train_operator=True,
    epochs=100, n_steps=25,
    work_dir="./runs/msd_mpc",
)
result = adalib.run_mpc(system=system, x0=[0.0, 0.0], options=options)

t, x, u = result.t, result.x, result.u
```

### 3c. MPC — autodiff gradients / batched CEM (built-in `cstr`, `triple_tank`)

The operator surrogate is a pure TF graph (MLP → panel weights W → linear
LPA basis), so the horizon-H rollout cost is exactly differentiable w.r.t.
the control sequence, and B candidate sequences can be evaluated in one
batched forward pass.

```python
import adalib

# Exact dJ/du via automatic differentiation through the operator (SLSQP jac)
result = adalib.run_mpc(
    system="triple_tank", x0=[190.0, 100.0, 140.0],
    options=adalib.MPCOptions(
        target={"h3": 150.0}, horizon=5, n_steps=20,
        gradient="autodiff",          # "fd" = same cost, finite differences
        work_dir="./runs/tt_autodiff_mpc",
    ),
)
print(result.metadata["opt_ms_per_step_mean"], result.metadata["opt_nfev_mean"])

# Sampling MPC exploiting batched surrogate inference (CEM or MPPI)
result = adalib.run_mpc(
    system="triple_tank", x0=[190.0, 100.0, 140.0],
    options=adalib.MPCOptions(
        target={"h3": 150.0}, horizon=5, n_steps=20,
        optimizer="CEM", cem_samples=512, cem_iters=8,   # or optimizer="MPPI"
        work_dir="./runs/tt_cem_mpc",
    ),
)
```

Supported for `cstr` and `triple_tank` (tracking) and `fedbatch_bioreactor`
(economic — same `gradient` / `optimizer` options with `mode="economic"`).
The generic `CallableODESystem` MPC path also accepts `gradient="autodiff"`,
using an analytic Jacobian through the pure-numpy LPA surrogate.
`gradient=None` + `optimizer="SLSQP"` (default) keeps the original loops.
See `examples/mpc/surrogate_mpc_showcase.py` for a head-to-head comparison
(FD vs autodiff vs CEM, plus a batch-throughput microbenchmark against
`solve_ivp`).

### 4. Inverse — parameter estimation from observations

```python
import adalib
import numpy as np
import tensorflow as tf

# --- Define system with unknown parameters ---------------------------------
def lv_rhs(t, x, u=None, p=None):
    prey, pred = x
    alpha, beta, gamma, delta = p["alpha"], p["beta"], p["gamma"], p["delta"]
    return [
        alpha * prey - beta * prey * pred,
        delta * prey * pred - gamma * pred,
    ]

def lv_rhs_tf(var_list, i, u=None, p=None):
    (prey, prey_t), (pred, pred_t) = var_list
    alpha = p["alpha"]; beta = p["beta"]
    gamma = p["gamma"]; delta = p["delta"]
    r1 = prey_t - (alpha * prey - beta * prey * pred)
    r2 = pred_t - (delta * prey * pred - gamma * pred)
    return tf.stack([r1, r2], axis=-1)

system = adalib.CallableODESystem(
    "lotka_volterra", lv_rhs, rhs_tf=lv_rhs_tf,
    state_names=["prey", "predator"],
)

# --- Generate synthetic observations ---------------------------------------
# (For a fair benchmark, sample observations from an independent solver such as
#  scipy.integrate.solve_ivp rather than from ADAlib's own forward solution;
#  see examples/inverse/lotka_volterra_inverse.py.)
true_p = {"alpha": 1.0, "beta": 0.1, "gamma": 1.5, "delta": 0.075}
ref    = adalib.run_forward(system, x0=[10.0, 5.0], t_span=(0.0, 15.0),
                            params=true_p,
                            options=adalib.ForwardOptions(n_seg=30))
data   = adalib.data_gen(ref, n_points=60, noise_std=0.05, seed=42)

# --- Set up inverse problem ------------------------------------------------
params = {
    "alpha": adalib.InverseParameter(0.5, lower=0.0, name="alpha"),
    "beta":  adalib.InverseParameter(0.2, lower=0.0, name="beta"),
    "gamma": 1.5,    # known — plain float
    "delta": 0.075,  # known — plain float
}

options = adalib.InverseOptions(
    n_seg=30, epochs=50, adam_lr=1e-3, adam_inner=100,
    lambda_data=10.0, lambda_physics=1.0,
    training_strategy="joint",
    normalize_data_loss=True,
)
result = adalib.run_inverse(system, x0=[10.0, 5.0], t_span=(0.0, 15.0),
                            params=params, data=data, options=options)

print(result.estimated_params)   # {"alpha": ..., "beta": ...}
result.plot(save_path="inverse_result.png")
result.plot_loss(save_path="inverse_loss.png")
result.plot_params(true_params=true_p, save_path="inverse_params.png")
```

---

## Plotting results

ADALib ships publication-quality plot helpers in `adalib.utils`.

```python
import matplotlib
matplotlib.use("Agg")   # headless / CI — call before importing adalib
import adalib

adalib.utils.set_adalib_plot_style()          # "sans" (default) or "serif"
```

### Forward result

```python
fig, axes = adalib.utils.plot_forward_result(
    result,
    reference=lambda t: np.exp(-t),    # callable, scipy OdeResult, or (t, y) tuple
    state_names=["$y$"],
    title="exponential decay",
    save_path="forward_result.png",
)
```

### Operator rollout (single or multi-case)

```python
fig, axes, metrics = adalib.utils.plot_operator_result(
    [r1, r2, r3],
    reference=[ref1, ref2, ref3],       # list of (t, y) tuples or scipy OdeResults
    state_names=["$C_A$", "$C_B$", "$T_R$", "$T_K$"],
    state_groups=[[0, 1], [2, 3]],      # optional grouped layout
    labels=["Case 1", "Case 2", "Case 3"],
    save_path="operator_result.png",
)
print(metrics["l2_rel"])   # shape: (n_cases, n_state)
```

### MPC closed-loop (single or multi-IC)

```python
fig, axes = adalib.utils.plot_mpc_result(
    all_results,                         # MPCResult or list of MPCResult
    state_names=["$C_A$", "$C_B$", "$T_R$", "$T_K$"],
    control_names=["$\\dot{Q}$"],
    target={"T_R": 136.0},               # dashed setpoint line
    labels=[f"IC {i+1}" for i in range(5)],
    save_path="mpc_result.png",
)
```

---

## Result inspection

Every workflow returns a result object with built-in convenience methods.

### ForwardResult

```python
result = adalib.run_forward(system, x0=[1.0], t_span=(0.0, 3.0))

t = result.t          # result.solution.t also works
y = result.y          # result.solution.y also works

result.plot(reference=lambda t: np.exp(-t), save_path="forward.png")
result.plot(reference="solve_ivp", save_path="forward_ref.png")

t, y = result.to_arrays()
result.save_npz("forward.npz")
print(result.list_artifacts())
```

### OperatorResult

```python
result = adalib.run_operator(system, x0=..., options=...)

result.plot(reference="solve_ivp", save_path="operator_rollout.png")
result.inference_plot(n_cases=4, save_path="operator_inference.png")

cases = result.infer(n_cases=4)   # list of {"t", "y_op", "y_ref", "u", "x0"}
result.save_inference({"t_0": cases[0]["t"], "y_0": cases[0]["y_op"]})
print(result.list_artifacts())
```

### MPCResult

```python
result = adalib.run_mpc(system, x0=..., options=...)

result.plot(save_path="mpc.png")
result.operator_inference_plot(n_cases=4, reference="solve_ivp",
                               save_path="mpc_surrogate.png")

t, x, u, cost = result.to_arrays()
result.save_npz("mpc.npz")
print(result.list_artifacts())
```

### InverseResult

```python
result = adalib.run_inverse(system, x0=..., t_span=...,
                            params=params, data=data, options=options)

print(result.estimated_params)       # {"alpha": 0.998, "beta": 0.102, ...}
print(result.runtime_sec)            # wall-clock time

# Trajectory
t, y = result.t, result.y           # (Nt_total,) and (n_state, Nt_total)

# Plots
result.plot(save_path="trajectory.png")           # recovered vs observed
result.plot_loss(save_path="loss.png")            # total / physics / data loss curves
result.plot_params(true_params={"alpha": 1.0},    # parameter convergence
                   save_path="params.png")

# History
print(result.loss_history)           # list of total loss per step
print(result.param_history)          # dict of list, one entry per log step

t, y = result.to_arrays()
result.save_npz("inverse.npz")
result.save_all("output/")           # trajectory + loss + params plots + npz
```

---

## InverseOptions reference

| Field | Default | Description |
|---|---|---|
| `basis` | `"adaf"` | Basis type (`"adaf"` only for inverse) |
| `n_seg` | `20` | Number of piecewise segments |
| `N_p` | `5` | ADA-F basis order per segment |
| `N_m` | `100` | ADA-F collocation points |
| `Nt_total` | `1000` | Total time-grid points for output |
| `gamma` | `0.8` | ADA-F decay parameter |
| `epochs` | `5` | Outer passes over all segments |
| `adam_inner` | `200` | Adam steps per segment per epoch |
| `adam_lr` | `1e-3` | Adam learning rate |
| `use_lbfgs` | `True` | L-BFGS polish after Adam |
| `lambda_physics` | `1.0` | Physics residual loss weight |
| `lambda_data` | `10.0` | Data fit loss weight |
| `training_strategy` | `"joint"` | `"joint"` (W+θ together) or `"alternating"` (W then θ) |
| `data_prefit_steps` | `0` | Adam steps fitting W only before joint training |
| `normalize_data_loss` | `True` | Scale data loss by number of observations |
| `normalize_physics_loss` | `False` | Scale physics loss by collocation count |
| `warm_seg_passes` | `1` | Extra passes on early segments |
| `n_warm_segs` | `3` | Number of early segments to warm-repeat |
| `true_params` | `None` | Ground-truth dict for convergence plots |
| `output_dir` | `None` | Auto-save plots/npz here if set |

**Training strategy guidance:**
- `"joint"` (default): works well when `lambda_data / lambda_physics ≤ 10`.
- `"alternating"`: recommended when the ratio exceeds ~100 (e.g. `lambda_data=500`), to prevent data loss from dominating physics gradients.

---

## InverseParameter reference

```python
# Unconstrained
p = adalib.InverseParameter(initial=0.5, name="alpha")

# Lower bound only (softplus transform)
p = adalib.InverseParameter(initial=0.5, lower=0.0, name="alpha")

# Box-constrained (sigmoid transform)
p = adalib.InverseParameter(initial=0.5, lower=0.0, upper=2.0, name="alpha")

# Read current estimate during/after training
print(p.numpy_value)       # Python float, post-transform
print(p.constrained)       # TF tensor used inside ODE
```

---

## Examples

Runnable scripts are in [`examples/`](examples/):

### Simple API (`examples/simple_api/`)

| Script | Description |
|---|---|
| `01_forward_example.py` | Forward — user-defined exponential decay |
| `02_forward_euler_example.py` | Forward — built-in Euler rigid body |
| `03_operator_example.py` | Operator — CSTR (3 ICs, scipy reference) |
| `04_mpc_example.py` | MPC — CSTR tracking (5 ICs) |
| `05_generic_tracking_mpc.py` | MPC — user-defined mass-spring-damper |

### Forward (`examples/forward/`)

| Script | Description |
|---|---|
| `lotka_volterra_forward.py` | Lotka-Volterra forward simulation |
| `euler_forward.py` | Euler rigid body forward simulation |

### Operator (`examples/operator/`)

| Script | Description |
|---|---|
| `train_bioreactor_operator.py` | Operator learning — fed-batch bioreactor |

### MPC (`examples/mpc/`)

| Script | Description |
|---|---|
| `cstr_tracking_mpc.py` | Tracking MPC — CSTR |
| `triple_tank_tracking_mpc.py` | Tracking MPC — triple tank |
| `bioreactor_economic_mpc.py` | Economic MPC — fed-batch bioreactor |
| `surrogate_mpc_showcase.py` | FD vs autodiff vs CEM comparison + batch-inference throughput benchmark |

### Inverse (`examples/inverse/`)

| Script | Description |
|---|---|
| `lotka_volterra_inverse.py` | Parameter estimation — Lotka-Volterra (α, β) |
| `euler_inverse.py` | Parameter estimation — Euler body inertia |

---

## Built-in systems

Inverse legend: ✅ validated · ⚠️ unverified · ❌ known to fail.

| Name | States | Operator | MPC | Inverse |
|---|---|:---:|:---:|:---:|
| `cstr` | C_A, C_B, T_R, T_K | ✅ | ✅ tracking | ⚠️ |
| `triple_tank` | h1, h2, h3 (cm; time in s) | ✅ | ✅ tracking | ⚠️ |
| `fedbatch_bioreactor` | Xs, Ss, Ps, Vs | ✅ | ✅ economic | ❌ |
| `lotka_volterra` | U, R (scaled prey, predator) | ✅ | — | ✅ |
| `lotka_volterra_ur` | r (prey), p (predator) | — | — | — |
| `euler` | ω₁, ω₂, ω₃ | — | — | ✅ |
| `damped_pendulum` | θ, ω | — | — | — |

```python
print(adalib.list_systems())
# ['cstr', 'damped_pendulum', 'euler', 'fedbatch_bioreactor',
#  'lotka_volterra', 'lotka_volterra_ur', 'triple_tank']
```

---

## Tests

`tests/` holds runnable scripts, one per feature. Most of them **train a model
when executed** (some for many minutes), so run them individually rather than
with a bare `pytest tests/`:

```bash
python tests/test_adalib_forward_euler.py     # one script
pytest -q tests/test_adalib_mpc_autodiff.py   # the fast pytest suite
```

| File | Coverage |
|---|---|
| `test_adalib_forward.py` | Forward — user-defined system |
| `test_adalib_forward_euler.py` | Forward — Euler rigid body |
| `test_adalib_forward_lotka.py` | Forward — Lotka–Volterra |
| `test_adalib_forward_pendulum.py` | Forward — damped pendulum |
| `test_adalib_forward_ev_itms.py` | Forward — additional user-defined system |
| `test_adalib_inverse_lv.py` | Inverse — Lotka–Volterra |
| `test_adalib_inverse_euler.py` | Inverse — Euler rigid body |
| `test_adalib_inverse_pendulum.py` | Inverse — damped pendulum |
| `test_adalib_operator.py` | Operator — CSTR (paper-size settings) |
| `test_adalib_operator_triple_tank.py` | Operator — triple tank (paper-size settings) |
| `test_adalib_operator_speaker.py` | Operator — user-defined system |
| `test_adalib_operator_ev_itms.py` | Operator — additional user-defined system |
| `test_adalib_mpc.py` | MPC workflow |
| `test_adalib_mpc_forward.py` | MPC with forward reference |
| `test_adalib_mpc_bioreactor.py` | Economic MPC — fed-batch bioreactor |
| `test_adalib_mpc_autodiff.py` | Autodiff / CEM surrogate MPC, gradient-vs-FD consistency (pytest) |

---

## Package layout

```
ADAF_Library/
├── adalib/                  # the installable package
│   ├── _vendor/legacy/      # vendored ADA-F / LPA / operator-MPC backend (required at runtime)
│   ├── systems/             # ODESystem, built-in systems, CallableODESystem, registry
│   ├── forward/             # ForwardSolver, ForwardOptions
│   ├── operator/            # OperatorLearner, predict_step, predict_rollout
│   ├── mpc/                 # MPCOptions, generic and surrogate MPC
│   ├── inverse/             # InverseSolver, InverseOptions, InverseParameter, data_gen
│   ├── workflows/           # run_forward, run_operator, run_mpc, run_inverse, run
│   └── utils/               # paths, metrics, plotting
├── examples/                # simple_api/, forward/, inverse/, operator/, mpc/
├── tests/                   # runnable feature scripts (see Tests)
├── docs/                    # Sphinx sources for the ReadTheDocs site
└── paper_artifacts/         # archived checkpoint + data for the paper's bioreactor results
```

---

## Reproducing the paper's reported results

The defaults used in the quick-start examples above are for illustration
only — they do **not** reproduce the accuracy reported in the paper. Each
benchmark's operator was trained with a *system-specific* network size and
training-set size (the paper's operator hyperparameter table, Appendix A):

| System | `N_p` | `n_seg` | `hidden` | `n_layers` | `n_train` | `epochs` | `batch_size` |
|---|---|---|---|---|---|---|---|
| Lotka–Volterra | 21 | 20 | 256 | 3 | 4,000 | 2,000 | 128 |
| Triple-tank | 20 | 10 | 128 | 3 | 20,000 | 2,000 | 128 |
| CSTR | 24 | 25 | 192 | 3 | 20,000 | 2,000 | 256 |
| Fed-batch bioreactor | 30 | 50 | 128 | 3 | 50,000 | 2,000 | 128 |

To reproduce, e.g., the CSTR operator accuracy, pass these explicitly rather
than relying on the quick-start defaults:

```python
import adalib

system  = adalib.get_system("cstr")
options = adalib.OperatorOptions(
    basis="lpa",
    n_train=20000, n_val=4000,
    hidden=192, n_layers=3,
    epochs=2000, batch_size=256,
    work_dir="./runs/cstr_operator_paper",
)
result = adalib.run_operator(
    system=system,
    x0=[0.8, 0.5, 134.14, 130.0],
    t_span=(0.0, 0.5),
    options=options,
)
```

`n_seg` comes from the built-in system's segmentation, not from
`OperatorOptions`. Complete runnable scripts at these sizes exist for the
CSTR (`tests/test_adalib_operator.py`) and the triple tank
(`tests/test_adalib_operator_triple_tank.py`). Training at these sizes takes
much longer than the quick-start defaults.

The archived checkpoint, training configuration, and evaluation data behind
the paper's fed-batch bioreactor results are in
`paper_artifacts/bioreactor_fig7_checkpoint/`; see that directory's README for
the exact reproduction steps. The current `FedBatchBioreactor` sampling ranges
differ from the archived run, so retraining with the row above does not
reproduce those numbers exactly.

**Forward simulation (Euler rigid body).** The paper's 1,500-parameter
configuration is `ForwardOptions(basis="adaf", N_p=10, n_seg=50, N_m=100,
Nt_total=2500, gamma=0.9, epochs=5, adam_inner=100, use_lbfgs=True)`; see also
`examples/forward/euler_forward.py`. The forward scripts in `tests/` use
smaller quick-start settings.

## Known limitations in v0.1.0

- **Generic operator learning** (`run_operator(CallableODESystem, …)`) trains
  **physics-residual only** — same philosophy as the built-in systems, no
  reference trajectory is generated or fit against (`adalib/mpc/
  _generic_mpc.py`: `_sample_inputs` + `_train_lpa_operator_physics`). The
  physics target is evaluated via `system.rhs()`; a fully-vectorized numpy
  call is attempted first (fast), falling back to a per-sample Python loop
  with a one-time warning if the RHS uses scalar-only ops (`float(u)`,
  `math.exp`, builtin `min`/`max` on arrays) — use `np.exp`/`np.minimum`/
  `np.maximum` instead for the fast path. Override the per-state residual
  normalization via `opts.res_scale={"state_name": scale}` if needed.
- **Generic economic MPC**: economic MPC is only supported for the built-in
  `fedbatch_bioreactor` system. Custom-system economic MPC is planned.
- **Generic tracking MPC still uses LPA Operator NN** (N_p=8, max_order=6)
  with **trajectory MSE loss** (data-driven, not physics residual) — unlike
  generic Operator learning above, this has not yet been migrated to the
  physics-only pipeline. Override hyperparameters via `opts.lpa_n_panels`,
  `opts.lpa_max_order`, `opts.lpa_nt_seg` if needed.
- **Thread safety**: operator and built-in MPC workflows use process-global
  configuration variables (`PROBLEM`, `BASIS`). Do not run two such workflows
  with different systems concurrently in the same process.
- **Generated artifacts** (datasets, checkpoints, plots) are written to
  `options.work_dir` and are not shipped with the package; each user generates
  them on first run.

---

## License

Released under the MIT License (see `LICENSE`). The vendored backend in
`adalib/_vendor/legacy/` is covered by `THIRD_PARTY_NOTICES.md`.
