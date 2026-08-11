# adalib-ode

**ADA-based ODE library** — forward solving, operator learning, model predictive control, and physics-informed inverse parameter estimation.

---

## Project status & handoff (updated 2026-07-16)

> This section is a running handoff log so a new maintainer can pick the project
> up without re-deriving the current state. The subsections below record **what
> works, what was decided, and what was deliberately dropped.**

### Inverse (parameter estimation) — scope frozen at Lotka–Volterra + Euler

The inverse solver is **validated only on the Lotka–Volterra (LV) and Euler
rigid-body systems** (low noise, full state observation). This is the intended,
final scope for the current paper/release — see the evidence below.

A robustness study (`scripts/inverse_robustness.py`, logs in
`runs/inverse_robustness_log.txt`) and a fed-batch bioreactor identifiability
study (`scripts/inverse_bio_robust.py`, `scripts/_bio_tune_check.py`) were run
on 2026-07-15. Findings:

| Scenario | ADA inverse | Classical NLS (Radau) baseline |
|---|---|---|
| LV, 0 % noise, full obs | ✅ α ≈ 0.04 %, γ ≈ 0.003 % rel-err | ✅ ~1e-11 |
| LV, 1 % noise | ⚠️ α ≈ 1 %, **γ ≈ 32 %** | ✅ α, γ < 0.2 % |
| LV, 3–5 % noise | ❌ γ rel-err 0.7–0.8 | ✅ < 0.7 % |
| LV, partial obs (prey only) | ❌ γ rel-err ≈ 1.0 | mixed |
| Fed-batch bioreactor Haldane (μ_max, K_S, K_I), **0 % noise** | ❌ rel-err 0.9–2.2 (fails) | ✅ ~1e-10 (perfect) |

**Decision (2026-07-15):** stop pursuing the bioreactor Haldane inverse. The
three Haldane kinetic constants are strongly correlated (poorly identifiable),
and ADA's joint W+θ optimization stalls near the initial guess while classical
NLS solves it exactly. Stronger settings (`_bio_tune_check.py`: more epochs,
data pre-fit, alternating optimizer) did **not** help — the `strong_prefit`
run actually got worse. The `alternating` variant was interrupted by a reboot
and never completed; it is **not** worth re-running given the two negative
results already in hand.

**Honest takeaway for the next maintainer:** ADA inverse is competitive with
classical NLS only for clean, fully-observed, well-conditioned problems. Under
measurement noise, partial observation, or strong parameter correlation it is
currently **not** competitive with NLS. Improving noise/identifiability
robustness is the natural next work item if inverse is to be extended beyond
LV/Euler.

### What changed on 2026-07-15 (new, uncommitted work)

New files added this session (not part of the original v0.1.0 snapshot):

- `scripts/inverse_robustness.py` — LV noise/n_obs/partial-observation sweep vs NLS.
- `scripts/inverse_bio_robust.py` — bioreactor Haldane identifiability study vs NLS.
- `scripts/_bio_tune_check.py`, `scripts/_bio_gate.py`, `scripts/_fhn_gate.py`,
  `scripts/_euler_gate.py` — fair-shot / gate diagnostic scripts.
- `scripts/pideeponet_benchmark.py`, `scripts/operator_speed_batched.py`,
  `scripts/tune_euler_forward.py` — operator accuracy/speed + Euler forward tuning.
- `runs/` — all logs and `results.json` outputs from the above (kept for the
  paper; regenerable, safe to delete).

Contextual notes from the session:
- **Euler forward tuning** (see `runs/euler_diag.txt`, `tune_euler_forward.py`):
  ADA-F Euler accuracy is controlled by `gamma` (sharp optimum ≈ 0.9), points
  per segment, and `n_seg` convergence.
- **LV inverse config drift:** `tests/test_adalib_inverse_lv.py` currently uses
  `lambda_data=1.0, training_strategy="joint", n_passes=1`. An earlier overnight
  note (`lv_overnight_summary.md`, 2026-07-06) recommended `lambda_data=500` +
  `alternating` for ~1 % error and warned `n_passes=2` caused drift to
  2.57 %/4.16 %. **These two configs disagree** — reconcile before quoting a
  single "official" LV inverse number. The paper currently reports the
  2.58 %/4.15 % (drift) numbers; verify against a fresh run before publishing.

### Paper revision log — abbreviation pass (`ADA_paper_final_before_abbreviation.tex` → `ADA_paper.tex`, 2026-07-20 → 2026-07-22)

We went through `ADA_paper.tex` section by section to tighten prose and cut
length. Below is what changed, organized by paper section. Purely cosmetic
wording trims are omitted; only substantive changes are listed.

> **Flag for review before submission:** the inverse-crime fix below (§2.3 /
> §4.2) was applied to the writeup but **not consistently to the underlying
> results** — see the callout under "ADA for Solving Inverse Problems." This
> is the one item that needs a decision (rerun the Euler experiment, or walk
> back the methodology claim) before the paper goes out.

**Global**
- Citation style switched from author–year (`natbib[authoryear]`) to numbered
  (`natbib[numbers,sort&compress]`, `unsrtnat`).
- 5 new citations added: `Kaipio2005`, `Hochreiter1998`, `DeepXDELVDemo`,
  `Kingma2014`, `Zhu1997`. None removed.
- 1 new figure added: LV operator schematic (`image14.png`, Appendix C).

**1. Introduction**
- PinnDE's backend description narrowed from "TensorFlow and JAX" to just
  "a JAX backend" — a factual correction about a competing library.
- PINN-limitations paragraph condensed; added `\citep{Hochreiter1998}` to
  support the "activation functions lose effectiveness after repeated
  differentiation" claim (previously uncited).
- ADA description clarified: accuracy parity with baselines is qualified as
  holding "when incorporated into PINNs," not as a standalone claim.
- **Dropped, not relocated:** the summary sentence claiming the solver
  "attains promising accuracy at millisecond-order batched inference,
  providing a competitive and physically interpretable alternative to
  PINN-based solvers" across four benchmark systems. Only the
  TensorFlow-backend sentence survived (moved to Appendix A).

**2. Methodology and Theory**
- *2.1 Anti-Derivative Approximator:* promoted from an unlabeled
  `\subsection*` to a proper numbered/labeled subsection; closing remark
  about the original ADA-F paper's residual-minimization framing dropped.
- *2.2 Feed-Forward Problems:* derivative-continuity equation for
  higher-order systems moved out to new **Appendix B**, with an added,
  more honest scope statement: *"none of the benchmark systems in this
  paper require m≥2, so this generalization is not exercised."* The
  warm-start explanation (reusing optimized panel weights as the next
  segment's initial weights) was cut with no replacement. Added citations
  for Adam (`Kingma2014`) and L-BFGS-B (`Zhu1997`).
- *2.3 Inverse Problems — most significant change:* observation data
  generation switched from **sampling ADA's own forward solution**
  (an inverse crime) to an **independent `scipy.integrate.solve_ivp`
  reference**, citing `Kaipio2005`.
  > ⚠️ **Only the prose was updated, not (verifiably) the results.** The
  > Euler rigid-body inverse still reports the identical converged values
  > (I₂=0.2999, I₃=0.3995) as before, its relative-error sentence was
  > deleted and replaced with a LaTeX comment: `% TODO: rerun Euler inverse
  > experiment with an independent reference integrator ... to avoid the
  > inverse crime, then update the converged I_2, I_3 values`. The
  > Lotka–Volterra numbers (α=38.97, γ=22.08, errors 2.58%/4.15%) are also
  > unchanged to the decimal despite the claimed protocol switch. Also
  > unresolved in both old and new versions: the Conclusion states recovery
  > was validated "across noise levels, observation densities, partial
  > observations, and initial guesses" — Section 4.2 reports none of that
  > sweep, in either version.
- *2.4 Operator Learning:* the concrete Lotka–Volterra operator derivation
  (equations + batched-input construction) moved to new **Appendix C**,
  which gains the new LV schematic figure. Swish-activation detail and the
  full training-loop description (Adam + cosine annealing + warm-up +
  gradient clipping) relocated to Appendix A.
- *2.5 Surrogate for MPC:* unchanged.

**3. ADA Solver User Implementation**
- Restructured: the old "Workflow" (7-step table) and "Extensions"
  subsections were removed from the main body and moved verbatim into an
  expanded **Appendix A** ("Software Workflow and User-Adjustable
  Settings"). No information lost, just relocated — Section 3 is now two
  short paragraphs plus a pointer to the appendix.

**4. Simulation Results**
- *4.1 Feed-Forward:* Euler/LV subsection order swapped (Euler now first).
  Euler numbers unchanged. **LV parameterization gap fixed** — added the
  explicit formula (α=2R, β=0.04RU, γ=1.06R, δ=0.02RU, U=200, R=20 ⇒
  α=40.0, β=160.0, γ=21.2, δ=80.0) that was previously only vaguely
  described, plus a new citation (`DeepXDELVDemo`).
- *4.2 Inverse:* see the §2.3 callout above — this is where the
  inconsistency actually surfaces in the results text.
- *4.3 Operator Learning:* per-state-error and timing tables merged per
  system (presentation only, no numbers changed). Triple-tank: added
  dataset-size disclosure (20k/4k/4k train/val/test trajectories). CSTR:
  removed explicit parameter-sampling ranges from main text (now just
  references `\citep{Fiedler2023}`). LV: **removed** the claim of being
  "~6× faster than cd-PINN" (`Li2025`) — no replacement. Bioreactor:
  archetype-scenario/per-state discussion moved to new **Appendix D**;
  **numeric correction** — the real-time speedup claim changed from
  "~two orders of magnitude" to **"~four orders of magnitude (~32,000×)"**
  faster than real process time (100 min), i.e. the old number looks like
  it was simply wrong.

**5. Control Application of OperatorADA**
- 5.1 Tracking MPC, 5.2 Bioreactor Economic MPC, 5.3 Differentiable/Batched
  Surrogate Inference: **no substantive changes** in any of the three
  subsections — text and all numbers are identical.

**6. Conclusion**
- Unchanged, including the pre-existing overclaim about inverse validation
  breadth noted under §2.3 above.

**Appendices A–D**
- All four are net-new or substantially expanded, built from content moved
  out of the main body (Appendix A: workflow/settings from old §3 + backend
  sentence from Intro + operator training details from §2.4; Appendix B:
  higher-order derivative continuity from §2.2, with new scope disclosure;
  Appendix C: LV operator construction from §2.4, with new figure;
  Appendix D: bioreactor archetype/error analysis from §4.3.4). No content
  was lost in these moves — only reorganized, with two exceptions
  called out above (dropped Intro summary sentence, dropped cd-PINN
  comparison).

### Environment

Use the conda **`tf`** env (adalib is installed editable there):

```bash
/home/jeongsulee/anaconda3/envs/tf/bin/python <script.py>
```

TensorFlow 2.21, GPU: RTX 4070 Ti. This is not a git repository — there is no
version control safety net; back up before large edits.

---

## Install

```bash
pip install adalib-ode
```

or from source:

```bash
pip install -e .
```

Requires Python ≥ 3.10, TensorFlow ≥ 2.13.

```python
import adalib   # distribution name is "adalib-ode"; import name remains "adalib"
```

---

## Feature support

| Feature | User-defined `CallableODESystem` | Built-in systems |
|---|:---:|:---:|
| Forward | ✅ Fully supported | ✅ Supported |
| Operator learning | ✅ Physics-residual only (LPA Operator NN) | ✅ Supported |
| MPC — tracking | ✅ LPA Operator NN surrogate | ✅ Supported |
| MPC — economic | ❌ Not yet supported | ✅ Supported |
| Inverse (parameter estimation) | ⚠️ Works; validated on LV/Euler only | ⚠️ Validated on LV/Euler only |

**Built-in systems:** `cstr`, `triple_tank`, `fedbatch_bioreactor`, `lotka_volterra`, `euler`

> **Inverse caveat:** the inverse solver is validated (tests + paper) only for
> `lotka_volterra` and `euler` under low noise / full observation. Inverse for
> `fedbatch_bioreactor` is **known to fail** (poorly identifiable Haldane
> kinetics); inverse for `cstr` / `triple_tank` is **unverified**. See
> *Project status & handoff* above.

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
(FD vs autodiff vs CEM + batch-throughput microbenchmark vs `solve_ivp`), and
`scripts/benchmark_surrogate_mpc.py` for the full paper benchmark (adds a
conventional `solve_ivp`-NMPC baseline, an $H$ sweep, and economic MPC).

### 4. Inverse — parameter estimation from observations

```python
import adalib
import numpy as np

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

Inverse legend: ✅ validated (tests + paper) · ⚠️ unverified · ❌ known to fail.

| Name | States | Operator | MPC | Inverse |
|---|---|:---:|:---:|:---:|
| `cstr` | CA, CB, TR, TK | ✅ | ✅ tracking | ⚠️ |
| `triple_tank` | h1, h2, h3 | ✅ | ✅ tracking | ⚠️ |
| `fedbatch_bioreactor` | Xs, Ss, Ps, Vs | ✅ | ✅ economic | ❌ |
| `lotka_volterra` | prey, predator | ✅ | — | ✅ |
| `euler` | ω₁, ω₂, ω₃ | — | — | ✅ |

```python
print(adalib.list_systems())
# ['cstr', 'euler', 'fedbatch_bioreactor', 'lotka_volterra', 'triple_tank']
```

---

## Tests

```bash
pytest -q tests/
python -X utf8 scripts/verify_merge_regression.py
```

Test coverage:

| File | Coverage |
|---|---|
| `test_adalib_forward.py` | Forward solver — generic systems |
| `test_adalib_forward_euler.py` | Forward — Euler rigid body |
| `test_adalib_forward_lotka.py` | Forward — Lotka-Volterra |
| `test_adalib_forward_pendulum.py` | Forward — pendulum |
| `test_adalib_operator.py` | Operator learning — generic |
| `test_adalib_operator_triple_tank.py` | Operator — triple tank |
| `test_adalib_operator_speaker.py` | Operator — speaker system |
| `test_adalib_mpc.py` | MPC workflow |
| `test_adalib_mpc_autodiff.py` | Autodiff/CEM surrogate MPC + gradient-vs-FD consistency |
| `test_adalib_mpc_forward.py` | MPC with forward reference |
| `test_adalib_mpc_bioreactor.py` | Economic MPC — bioreactor |
| `test_adalib_inverse_lv.py` | Inverse — Lotka-Volterra |
| `test_adalib_inverse_euler.py` | Inverse — Euler body |
| `test_adalib_inverse_pendulum.py` | Inverse — pendulum |

---

## Package layout

```
adalib_project/
├── adalib/                         # public API (pip-installable)
│   ├── _vendor/legacy/             # vendored ADA-F / LPA / Operator-MPC backend
│   ├── systems/                    # ODESystem, 5 built-ins, CallableODESystem, registry
│   ├── forward/                    # ForwardSolver, ForwardOptions
│   ├── operator/                   # OperatorLearner, predict_step, predict_rollout
│   ├── mpc/                        # MPCOptions, generic tracking MPC
│   ├── inverse/                    # InverseSolver, InverseOptions, InverseResult,
│   │                               #   InverseParameter, ObservationData, data_gen
│   ├── workflows/                  # run_forward, run_operator, run_mpc, run_inverse, run
│   └── utils/                      # paths, metrics, plotting
├── examples/
│   ├── simple_api/                 # numbered quickstart scripts (01–05)
│   ├── forward/                    # forward examples
│   ├── operator/                   # operator learning examples
│   ├── mpc/                        # MPC examples
│   └── inverse/                    # inverse parameter estimation examples
├── tests/                          # pytest test suite (13 test files)
└── scripts/verify_merge_regression.py
```

---

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
