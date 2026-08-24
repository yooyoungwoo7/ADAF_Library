# Lotka-Volterra Inverse — Overnight Results (2026-07-06)

## TL;DR

**Both parameters recovered within ~1%** using alpha/gamma parameterization.
`alpha` error: **1.06%**, `gamma` error: **0.96%**.

---

## What Was Fixed

### 1. Parameter Collapse (alpha → 0, gamma → 0)

**Root cause**: Joint W+θ Adam optimization let W absorb the physics signal,
giving θ zero gradient. Also joint L-BFGS was finding degenerate solutions (alpha=0).

**Fix**: Alternating optimizer in `InverseSolver`:
- W-step: optimize W with phys+data loss (θ frozen)
- θ-step: optimize θ with phys loss only (W frozen)
- L-BFGS Stage A: W-only (maxiter=200)
- L-BFGS Stage B: θ-only (maxiter=50, conservative)

### 2. U/R Parameterization Failure

**Root cause**: ∂L_phys/∂U involves r·p terms that change sign over the LV oscillation period
→ net gradient ≈ 0.

**Fix**: Reverted to alpha/gamma parameterization:
- ∂L_phys/∂alpha ∝ (α_true − α) · mean(r²) — always same sign
- ∂L_phys/∂gamma ∝ (γ_true − γ) · mean(p²) — always same sign

### 3. lambda_data Too Low

**Root cause**: With lambda_data=1, W compromises between wrong physics and true data
→ W doesn't represent the true trajectory → noisy θ gradient.

**Fix**: lambda_data=500 → W strongly fits observed data → clean θ gradient.

---

## Hyperparameter Search Results (17 configs)

| Config | λ_data | N_m | n_seg | passes | α err | γ err | time |
|--------|--------|-----|-------|--------|-------|-------|------|
| **L4** | **500** | **20** | **25** | **1** | **1.06%** | **0.96%** | **207s** |
| L3     | 100    | 20  | 25    | 1      | 6.56% | 3.80% | 207s |
| R2     | 100    | 10  | 25    | 1      | 4.58% | 11.3% | 190s |
| E1     | 100    | 10  | 25    | 1      | 3.81% | 12.5% | 578s |
| M2     | 50     | 20  | 25    | 1      | 99.8% | 99.5% | (collapsed) |

**lambda_data=500 was the decisive factor.**

---

## Final Config (in both test files)

```python
adalib.InverseOptions(
    n_seg=25,         # covers ~4.5 oscillation cycles
    N_p=5,
    N_m=20,           # Nt_seg=200 >> N_m=20 → overdetermined physics
    Nt_total=5000,
    lambda_physics=1.0,
    lambda_data=500.0,
    epochs=3,
    adam_inner=100,
    adam_lr=1e-3,
    use_lbfgs=True,
    n_passes=1,       # 1 pass; n_passes=2 caused drift (2.57%/4.16%)
    dtype="float64",
)
```

**Note**: n_passes=2 with more epochs made results WORSE.
The first pass with the right hyperparameters is sufficient and cleaner.

---

## Files Updated

- `adalib_project/tests/test_adalib_inverse_lv.py` — best config, alpha/gamma
- `C:\Users\young\Desktop\adalib_test\test_adalib_inverse_lv.py` — same (was U/R, now fixed)

## Key Modified Source Files

- `adalib/inverse/solver.py` — alternating optimizer (W-step / θ-step)
- `adalib/systems/lotka_volterra.py` — added `LotkaVolterraUR` class
- `adalib/systems/registry.py` — registered "lotka_volterra_ur"
