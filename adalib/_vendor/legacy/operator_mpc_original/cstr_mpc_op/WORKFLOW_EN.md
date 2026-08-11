# Operator Library — Workflow (English)

A unified physics-informed neural-operator library for ODE systems with a
selectable **basis** (LPA / ADA-F) and **problem** (4 bundled). One codebase
handles every (problem × basis) combination; switch via `config.py`, env
vars, or CLI flags.

## 1. Directory layout

```
operator_lib/
├── config.py                    # PROBLEM_NAME, BASIS_NAME + PROBLEM_CONFIGS
├── main_train.py                # universal training entry point
├── main_eval.py                 # universal eval / inference entry point
├── run_all.py                   # batch driver (sequential or --parallel N)
├── requirements.txt
├── problems/
│   ├── base_problem.py          # abstract Problem interface
│   ├── lotka_problem.py         # Lotka-Volterra (Hamiltonian conservation)
│   ├── bioreactor_problem.py    # fed-batch Haldane (output_scale)
│   ├── cstr_problem.py          # CSTR (Arrhenius + heat balance, output_scale)
│   ├── triple_tank_problem.py   # Torricelli triple tank
│   └── registry.py              # name → instance
├── models/
│   ├── basis.py                 # factory: get_basis_cls("lpa"|"adaf")
│   ├── lpa_basis.py             # Legendre panel basis, enforces x(0)=x0
│   ├── adaf_basis.py            # Fourier basis, enforces x(0)=x0 AND ẋ(0)=ẋ0
│   ├── operator_net.py          # vanilla MLP (zero-init, optional output_scale)
│   └── learner.py               # generic OperatorLearner(problem, basis_name=...)
├── data/
│   └── dataset_builder.py       # generic builder + loader
├── utils/                       # _style, io, metrics, poly, plotting, sweep_plots
├── data_files/<problem>/        # per-problem RK45/BDF + segment npz
└── results/<problem>/           # per-problem training outputs
```

## 2. Registered problems × bases

| name           | system                              | states / params | solver | output_scale |
|----------------|-------------------------------------|-----------------|--------|--------------|
| `lotka`        | Lotka-Volterra (U, R) family        | 2 / 4           | RK45   | — (uniform RES_SCALE) |
| `bioreactor`   | Fed-batch (Haldane kinetics)        | 4 / 3           | RK45   | ✓ `(0.033, 3.4, 0.025, 0.1)` |
| `cstr`         | CSTR (Arrhenius + heat balance)     | 4 / 4           | BDF    | ✓ `(0.5, 0.3, 30, 15)` |
| `triple_tank`  | Torricelli triple tank              | 3 / 2           | RK45   | — (uniform RES_SCALE) |

| basis  | hard IC                          | network output | notes |
|--------|----------------------------------|---------------|-------|
| `lpa`  | `x(0) = x0`                      | `(B, state_dim, LPA_N_P)` panel weights | default; W → series → analytic integration |
| `adaf` | `x(0) = x0` AND `ẋ(0) = f(x0,θ)` | `(B, state_dim, ADAF_N_P)` panel weights | `(ξ+1)` shift makes both free parts vanish at `t=0` |

Switch via `config.PROBLEM_NAME`/`BASIS_NAME`, env `PROBLEM=`/`BASIS=`, or
CLI `--problem`/`--basis`.

## 3. Design philosophy — single-segment training + inference-only rollout

```
[Training]   Learn one T_seg-long segment across diverse ICs.
             loss = (ẋ_pred − f(x_pred, θ))² / RES_SCALE²   (+ cons_w · H term)
             x(0) = x_0 is structurally enforced by both bases (and ẋ(0) by ADA-F).

[Inference]  Chain N_SEG segments — feed seg-k's x_end as seg-(k+1)'s x_0.
             No gradient flow.
```

K-segment chained-gradient training (`fit_rollout`) is **diagnostic only** —
enabled explicitly with `--rollout_epochs N`. Default `ROLLOUT_EPOCHS=0`.

### Physics-informed input augmentation (optional)
Each problem can override `derived_features_tf(x_input)` to inject extra
physics-aware columns into the network input. The vanilla `OperatorNet`
auto-detects these and concatenates `raw + derived features`. All four
bundled problems register 6 each:

| problem | 6 derived features |
|---|---|
| `lotka` | `ln(r₀/r*)`, `ln(p₀/p*)`, `V₀` (Hamiltonian), `ω·T_seg`, `α − β p₀`, `δ r₀ − γ` |
| `bioreactor` | `μ(Ss₀)`, `D₀`, `(μ−D)Xs₀`, `μSₘ + D(S_in−Ss₀)`, `Ss₀/K_I`, `Vs₀ + inp·DT_SEG` |
| `cstr` | `k1`, `k2`, `k3` (Arrhenius at T_R₀), `1/F`, `T_R − T_K`, reaction heat at IC |
| `triple_tank` | `Q13`, `Q32`, `Q20` at IC, plus per-tank net inflows |

If a problem doesn't override (`n_derived_features=0`), the network is a
plain MLP on the raw input.

### Per-state output scaling (optional)
A problem with non-uniform `RES_SCALE` can set `output_scale = RES_SCALE` to
scale W per state. This balances `∂Loss/∂W` across states; without it,
problems like CSTR (3,600× imbalance) and bioreactor (18,000×) waste epochs
calibrating per-state magnitudes. Uniform-RES_SCALE problems leave it `None`
since Adam absorbs constant scalings.

### Zero-init last layer
The OperatorNet's projection to W is zero-initialised so the basis starts
from a physically-valid baseline:
- LPA → `x(t) = x0` (identity)
- ADA-F → `x(t) = x0 + ẋ0·t` (affine free-part only)

Avoids initial residual blow-up on stiff problems (e.g., CSTR Arrhenius
`exp()` overflow when `s₁ ≈ 100` would amplify random initial weights).

## 4. Training workflow

```
main_train.py [--problem <name>] [--basis lpa|adaf]
   │
   ├─ get_problem(PROBLEM_NAME)
   │
   ├─ _ensure_datasets(problem, paths, seed)
   │    ├─ build_and_save_fullcase   ← problem.sample_cases + RK45/BDF
   │    └─ build_segments_from_fullcase
   │         (1 random seg/case OR all N_SEG/case based on problem hook)
   │
   ├─ OperatorLearner(problem, hidden, n_layers, x_mean, x_std,
   │                  basis_name=BASIS_NAME)
   │    ├─ OperatorNet  (vanilla MLP, zero-init, optional output_scale)
   │    ├─ build_basis(BASIS_NAME)         : LPA or ADA-F
   │    ├─ xdot0 = problem.rhs_tf(x0, θ)   : ADA-F's ẋ(0) hard-IC anchor
   │    └─ (residual + cons_w·conservation) loss
   │
   ├─ fit(...)  [single-segment physics-only training]
   │    ├─ train_step  @tf.function(jit_compile=True)
   │    ├─ cosine warmup LR (PHYSICS_LR → LR_MIN)
   │    ├─ tqdm progress bar
   │    └─ best-val checkpoint: epoch_XXXX_best.weights.h5
   │
   │  (optional: --rollout_epochs N → fit_rollout(...) diagnostic K-segment)
   │
   └─ save config_snapshot.json (records basis, hidden, N_p, etc.),
            train_summary.json, history.png
```

### Run

```bash
USE_GPU=0 python main_train.py                                  # config defaults
USE_GPU=0 python main_train.py --problem cstr                   # problem only
USE_GPU=0 python main_train.py --problem cstr --basis adaf      # basis switch
PROBLEM=triple_tank BASIS=adaf USE_GPU=0 python main_train.py   # env override

# Batch drive all 4 problems
USE_GPU=0 python run_all.py                                     # sequential
USE_GPU=0 python run_all.py --parallel 4                        # 4 in parallel
USE_GPU=0 python run_all.py --basis adaf                        # all on ADA-F
```

Key CLI args: `--problem`, `--basis`, `--epochs`, `--batch_size`, `--lr`,
`--hidden`, `--n_layers`, `--rollout_epochs`, `--rebuild_data`, `--seed`.

## 5. Inference / validation workflow

```
main_eval.py --problem <name> --weights <ckpt> --mode {validation|custom}
   │
   ├─ _build_learner  : auto-loads arch + basis from config_snapshot.json
   │                    (forces inference to use the basis trained against)
   │
   ├─ validation
   │    ├─ _rollout_and_report : chain N_SEG segments, inference-only
   │    │    → *_traj.png, *_residual.png, *_rollout.npz
   │    ├─ plot_val_comparison : N×state grid (val data)
   │    ├─ plot_param_sweeps   : auto from problem.sweep_specs()
   │    └─ plot_random_cases   : 4 cases via problem.random_input()
   │
   └─ custom: --x_input <input_dim floats>
```

### Run

```bash
USE_GPU=0 python main_eval.py --problem lotka \
    --weights results/lotka/<run>/checkpoints/<best>.weights.h5 \
    --mode validation --n_cases 5 --comparison_n 4 --sweep_n 8 --random_n 4
```

## 6. Adding a new problem

1. Create `problems/<name>_problem.py` extending `BaseProblem`. Required:
   - `sample_cases(n, seed)` → `(x0, theta, meta_dict)`
   - `rhs_np(t, x, theta)` (numpy)
   - `rhs_tf(x, theta)` (TF, broadcasting)
   Optional hooks:
   - `conservation_quantity(x, theta)` + `cons_w > 0`
   - `derived_features_tf(x_input)` + `n_derived_features` +
     `derived_mean / derived_std`
   - `output_scale = RES_SCALE` when RES_SCALE is non-uniform across states
   - `diverse_random_inputs(n, rng)` for hand-picked archetypes
   - `extra_plot_traces()` for auxiliary rows below state rows
   - `segment_sampling_strategy = "all"` to build every segment per case
   - `apply_train_oversampling(seg)` for row-level duplication at load time
   - plot metadata: `nominal_input()`, `sweep_specs()`, `case_subtitle()`,
     `state_units()`, `state_plot_labels()`, `random_input()`, `time_factor`
2. Register in `problems/registry.py`.
3. Add an entry in `config.PROBLEM_CONFIGS`.

No other code needs to change.

## 7. Key design points

- **PROBLEM × BASIS single entry point** — no codebase forking. New problem =
  one `problems/` file + one `config` entry + one `registry` line.
- **Both bases hard-enforce IC**: LPA enforces `x(0)=x0`, ADA-F enforces both
  `x(0)=x0` and `ẋ(0)=f(x0,θ)`. Network only predicts panel weights; basis
  composition (Legendre or Fourier integration) is in dedicated code.
- **Single-segment + inference-only rollout** is canonical. K-segment chained
  gradient training is diagnostic only.
- **Physics-only loss + optional conservation term**. Label-free.
- **JIT-compiled** every training step.
- **Three-tier scaling system**:
  1. Input normalisation (`x_mean`, `x_std`, OperatorNet)
  2. Output scaling (`output_scale`, OperatorNet, optional)
  3. Loss residual (`RES_SCALE`, config)
  Non-uniform RES_SCALE problems set `output_scale = RES_SCALE` to balance
  gradient magnitudes.
- **Zero-init last layer**: training starts with W=0 → physically-valid
  baseline (`x=x0` for LPA, `x=x0+ẋ0·t` for ADA-F) → avoids initial NaN on
  stiff problems.
