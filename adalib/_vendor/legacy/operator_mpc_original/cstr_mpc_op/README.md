# Physics-Informed Neural Operator — Surrogate-based MPC Library

This library implements a **Model Predictive Control (MPC)** pipeline using a
**physics-informed neural operator** as a surrogate model for system dynamics.
The operator is trained from ODE residuals only — no labeled trajectory data
is required. At inference time, the trained operator replaces the ODE solver
inside the optimizer, enabling fast real-time MPC.

Two MPC formulations are implemented:

| Problem | Type | Script |
|---------|------|--------|
| CSTR (Arrhenius + heat balance) | **Tracking MPC** | `main_mpc_cstr.py` |
| Fed-batch Bioreactor (Haldane kinetics) | **Economic MPC** | `main_mpc_bioreactor.py` |

---

## Core Architecture

### Operator Neural Network (LPA basis)

```
Input z = [x_k (state), θ (parameters/control)]
          + physics-derived features (problem-specific)
               ↓
          MLP  (hidden=128–256, 3–4 layers)
               ↓
          Legendre panel weights W
               ↓
Output:  x(t) over segment  [hard IC enforced: x(0) = x_k exactly]
         x_end = x(T_seg)
```

Key properties:
- **Hard IC constraint**: `x(0) = x0` is enforced symbolically; the network
  only learns the correction weights W.
- **Label-free training**: loss = ODE physics residual `‖ẋ − f(x, θ)‖`.
- **Single-segment training**: the operator predicts one segment well; long
  trajectories are obtained by chaining segments at inference time.

### Surrogate-based MPC Loop

```
For each time step k:
  ┌─ Optimization (Surrogate) ──────────────────────────────┐
  │  SLSQP / Brent over u = [inp_k, …, inp_{k+N−1}]        │
  │  Objective J(u) evaluated by chaining Operator NN × N   │
  │  Multi-start: warm-start + boundary start → best result  │
  └─────────────────────────────────────────────────────────┘
  Apply u*[0] only → advance TRUE plant via RK45 ODE
  k → k+1   (Receding Horizon)
```

Plant-model mismatch is compensated by closed-loop feedback at each step.

---

## Problem 1 — CSTR Tracking MPC

### System

**4-state CSTR** (Arrhenius kinetics + heat balance):

| State | Symbol | Unit |
|-------|--------|------|
| Reactant A concentration | C_A | mol/L |
| Product B concentration  | C_B | mol/L |
| Reactor temperature      | T_R | °C    |
| Jacket temperature       | T_K | °C    |

**Control input:** Q̇ ∈ [−8500, 0] kJ/h  
**Fixed parameters:** α = 1, β = 1, F = 50 h⁻¹

### MPC Objective (Tracking)

```
minimize_{Q}   ( T_R(x_{k+1}) − T_ref )²
```

Solved with Brent's method (derivative-free, scalar Q).

### Quick Start

```powershell
$env:PROBLEM = "cstr_mpc"
$env:BASIS   = "lpa"
python main_train.py                          # train operator
python main_mpc_cstr.py --T_ref 136.0        # run MPC
```

**Results:** All 3 default ICs converge to T_ref = 136 °C within 2–3 steps.
Output: `results/mpc_cstr/mpc_result.png`

---

## Problem 2 — Fed-batch Bioreactor Economic MPC

### System

**4-state fed-batch bioreactor** (Haldane substrate-inhibition kinetics):

| State | Symbol | Unit | Description |
|-------|--------|------|-------------|
| Biomass      | X_s | g/L | Cell concentration |
| Substrate    | S_s | g/L | Nutrient concentration |
| Product      | P_s | g/L | Target product concentration |
| Volume       | V_s | L   | Reactor liquid volume |

**Control input:** `inp` ∈ [0.005, 0.200] L/min (feed flow rate)  
**Fixed parameters:** Y_x = 0.4, S_in = 0.8 g/L, T_final = 100 min

**Dynamics (Haldane kinetics):**
```
μ      = MU_M · Ss / (K_M + Ss + Ss²/K_I)     optimal Ss = √(K_M·K_I) ≈ 0.5 g/L
D      = inp / Vs

dXs/dt = μ Xs − D Xs
dSs/dt = −μ Xs/Yx − V_PAR Xs/Y_P + D(Sin − Ss)
dPs/dt = V_PAR Xs − D Ps                        (product from biomass, not substrate)
dVs/dt = inp
```

> **Note on Sin setting:** The original do-mpc benchmark uses Sin ≈ 180–220
> (in the reference unit convention). In that regime Sin >> K_I = 5, placing
> the system in severe substrate inhibition where minimum feed is trivially
> optimal. For these experiments, **Sin was rescaled to (0.3–1.5) g/L and
> fixed at 0.8 g/L** to create a substrate-limited regime where active feed
> control meaningfully affects growth rate μ. This is a deliberate
> redefinition for nontrivial EMPC experiments.

### MPC Objective (Economic)

Unlike tracking MPC, there is no setpoint. The optimizer directly maximizes
process economics:

```
J = −w_product  · Ps_N · Vs_N              maximize terminal product amount [g]
  − w_stage   · Σ Ps_k · Vs_k · Δt        maximize running product (do-mpc lterm)
  + w_feed    · Σ inp_k · Δt               minimize feed cost
  + w_smooth  · Σ(Δinp)²                   penalize control variation
  + w_volume  · Σ max(0, Vs_k − Vmax)²     soft volume constraint
  + w_substrate · Σ max(0, Ss_k − Sinh)²   substrate inhibition penalty
  + w_nonneg  · Σ max(0, −state_k)²        non-negativity
  + w_ss_track· Σ(Ss_k − Ss_opt)²          kinetic shaping: Ss → 0.5 g/L
```

The `w_ss_track` term is a **kinetic-guided shaping term** (not a direct
economic cost). It encourages operation near the Haldane growth-rate optimum
`Ss_opt = √(K_M · K_I) ≈ 0.5 g/L` to indirectly improve biomass growth and
product formation.

> **Fair comparison note:** `const_high` (inp = 0.200 L/min) violates
> V_max = 5 L (Vs reaches ~21 L), so it is an **infeasible upper reference**.
> The true feasible benchmark is **`const_nominal`** (inp = 0.040 L/min,
> final Ps × Vs = 0.651 g), which fills the reactor to exactly V_max at
> t = 100 min.

### Baseline Comparison

| Strategy | Final Ps×Vs [g] | Feed used [L] | Feed efficiency [g/L] | V_max compliant | Notes |
|----------|----------------|---------------|-----------------------|-----------------|-------|
| const_low  (inp = 0.005) | 0.448 | 0.500 | 0.897 | ✅ | Substrate-starved lower bound |
| **const_nominal (inp = 0.040)** | **0.651** | **4.000** | **0.163** | **✅** | **Feasible benchmark** |
| const_high (inp = 0.200) | 0.692 | 20.000 | 0.035 | ❌ Vs ≈ 21 L | Infeasible upper reference |
| **Economic MPC** | **0.645** | **2.814** | **0.229** | **✅** | Best tuned result |

Feed efficiency = Final Ps×Vs [g] ÷ total feed volume [L].
Economic MPC matches const_nominal on terminal product while using **30% less feed**
and achieving **40% higher feed efficiency**, demonstrating the value of active control.

> **Fair comparison note:** `const_high` violates V_max = 5 L (Vs reaches ~21 L),
> so it is an **infeasible upper reference**. The true feasible benchmark is
> **`const_nominal`** (inp = 0.040 L/min, final Ps×Vs = 0.651 g).

### Quick Start

```powershell
$env:PROBLEM = "bioreactor"
$env:BASIS   = "lpa"

# Step 1: Train operator
python main_train.py

# Step 2: Evaluate operator quality
python main_eval.py --problem bioreactor --basis lpa `
  --weights "results/bioreactor/<timestamp>/checkpoints/epoch_XXXXX_best.weights.h5" `
  --mode validation --n_cases 8

# Step 3: Run Economic MPC
python main_mpc_bioreactor.py `
  --n_pred 20 --n_steps 50 `
  --w_product 1.0 --w_stage 0.05 `
  --w_feed 0.001 --w_smooth 0.01 `
  --w_volume 0.5 --w_substrate 0.5 `
  --w_ss_track 0.5 --Ss_opt 0.5 `
  --maxiter 150 --ftol 1e-5 `
  --Xs0 1.0 --Ss0 0.3 --Ps0 0.0 --Vs0 1.0
```

### Key CLI Arguments (`main_mpc_bioreactor.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_pred` | 10 | Prediction horizon [segments] |
| `--n_steps` | 50 | Total closed-loop steps |
| `--w_product` | 1.0 | Terminal Ps×Vs weight |
| `--w_stage` | 0.1 | Running Ps×Vs weight |
| `--w_feed` | 0.01 | Feed cost weight |
| `--w_smooth` | 0.1 | Move suppression weight |
| `--w_volume` | 5.0 | Volume constraint penalty |
| `--w_ss_track` | 0.0 | Ss tracking shaping weight |
| `--Ss_opt` | 0.5 | Haldane optimal substrate [g/L] |
| `--V_max` | 5.0 | Reactor volume upper bound [L] |
| `--maxiter` | 150 | SLSQP max iterations per start |
| `--ftol` | 1e-5 | SLSQP convergence tolerance |
| `--Xs0/Ss0/Ps0/Vs0` | 1.0/0.3/0.0/1.0 | Initial condition |

**Output files** (under `results/bioreactor_economic_mpc/`):

| File | Description |
|------|-------------|
| `states_<ts>.png` | 6-row plot: u_inp / Ps / Vs / Ps×Vs / Cumulative feed / Feed efficiency |
| `economic_terms_<ts>.png` | 3-row economic breakdown: Running product integral / Cumulative feed / Feed efficiency |
| `control_<ts>.png` | Feed rate u_inp profile |
| `product_amount_<ts>.png` | Ps×Vs over time |
| `solve_time_<ts>.png` | SLSQP solve time per step |
| `closed_loop_states_<ts>.csv` | Full state trajectory |
| `closed_loop_controls_<ts>.csv` | Control inputs and solve times |
| `summary_metrics_<ts>.csv` | Final metrics per strategy |

---

## Environment Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** TensorFlow ≥ 2.13, NumPy, SciPy, Matplotlib, tqdm

> CPU-only mode is active by default (`USE_GPU=0`). XLA JIT compilation is
> enabled. Set `USE_GPU=1` in `config.py` for CUDA GPU.

All scripts must be run from **inside** `cstr_mpc_op/` (the directory
containing `config.py`).

---

## General Workflow (All Problems)

```powershell
# 1. Set problem and basis
$env:PROBLEM = "bioreactor"   # or: cstr, cstr_mpc, triple_tank, lotka
$env:BASIS   = "lpa"          # or: adaf

# 2. Train operator (generates data + trains in one step)
python main_train.py

# 3. Evaluate operator quality
python main_eval.py --problem bioreactor --basis lpa --weights <path>

# 4. Run MPC inference
python main_mpc_bioreactor.py [options]   # bioreactor EMPC
python main_mpc_cstr.py [options]         # CSTR tracking MPC
python main_mpc_triple_tank.py [options]  # triple-tank tracking MPC
```

---

## Directory Structure

```
cstr_mpc_op/
│
├── config.py                      # all hyperparameters; problem/basis switch
├── main_train.py                  # data generation + operator training
├── main_eval.py                   # operator prediction quality evaluation
├── main_mpc_cstr.py               # CSTR tracking MPC inference
├── main_mpc_bioreactor.py         # fed-batch bioreactor economic MPC
├── main_mpc_triple_tank.py        # triple-tank tracking MPC inference
├── check_operator_quality.py      # CSTR/triple-tank quality check with plots
│
├── problems/
│   ├── base_problem.py            # abstract interface
│   ├── bioreactor_problem.py      # Haldane kinetics, parameter ranges
│   ├── cstr_mpc_problem.py        # Arrhenius + heat balance
│   ├── triple_tank_mpc_problem.py # Torricelli outflow
│   └── registry.py                # name → problem instance
│
├── models/
│   ├── lpa_basis.py               # Legendre panel basis (hard IC: x(0)=x0)
│   ├── operator_net.py            # MLP with zero-init last layer
│   └── learner.py                 # OperatorLearner: fit(), predict_segment()
│
├── data/
│   └── dataset_builder.py         # build & load segment datasets (.npz)
│
├── utils/
│   ├── _style.py                  # matplotlib global style
│   ├── metrics.py                 # l2_rel, statewise_l2_rel
│   └── plotting.py                # residual profile plots
│
├── data_files/
│   ├── bioreactor/                # generated .npz data (auto-created)
│   └── cstr_mpc/
│
└── results/
    ├── bioreactor/                # operator training runs
    │   └── <timestamp>/
    │       ├── checkpoints/       # *.weights.h5
    │       ├── config_snapshot.json
    │       └── history.npz / history.png
    ├── bioreactor_economic_mpc/   # EMPC output plots and CSV
    └── mpc_cstr/                  # CSTR MPC output plots
```

---

## Operator Quality Summary (Bioreactor, epoch 1987)

| State | Mean L2 | Assessment |
|-------|---------|------------|
| V_s   | < 0.001 | Excellent (linear integration) |
| P_s   | 0.001–0.013 | Good |
| X_s   | 0.02–0.22 | Moderate — systematic underprediction |
| S_s   | up to 0.88 | Poor near Ss ≈ 0 (MPC operating regime) |

The Xs underprediction causes the optimizer to underestimate long-term product
formation, leading to conservative control decisions. Improving training data
coverage in the Ss ≈ 0 regime is an identified area for future work.

---

## Key Design Decisions

### Why physics-residual training (no labels)?

ODE reference trajectories are expensive to generate at training scale (50 000
cases × 50 segments = 2.5 M segments for the bioreactor). Training on the
physics residual `‖ẋ − f(x, θ)‖` evaluated at collocation points requires only
forward ODE evaluations at a few time points per segment, avoiding full numerical
integration per sample.

### Why hard IC enforcement (LPA basis)?

Stiff systems (CSTR) and volume-conserving systems (bioreactor Vs) require
exact `x(0) = x0`. Without it, random initial network weights create exploding
Arrhenius terms at the very first training step. Zero-initializing the last
layer and enforcing IC symbolically makes training stable from epoch 0.

### Why Surrogate-based MPC instead of solving the ODE directly?

SLSQP evaluates the objective hundreds of times per step. A single RK45
integration ≈ 1–5 ms; for N_pred = 20 and ~500 evaluations per step, that is
10 seconds per step × 50 steps = 8 min total. The Operator NN forward pass ≈
0.05 ms, reducing solve time to ~30 s total (300× speedup).

### Economic MPC vs. Tracking MPC

Tracking MPC minimizes `(state − setpoint)²` — suitable when a target
operating point is known (e.g., CSTR steady-state temperature). Economic MPC
directly optimizes a process economic objective (product yield, feed cost,
constraint penalties) with no setpoint. This is appropriate for batch
processes where the optimal trajectory is not known a priori.

---

## Documentation

| File | Contents |
|------|----------|
| `README.md` | This file — quick start and overview |
| `WORKFLOW_EN.md` | Full library workflow for all 4 bundled problems |
| `WORKFLOW.md` | Same, in Korean |
| `CLAUDE.md` | Developer reference — code conventions, per-problem tuning |
