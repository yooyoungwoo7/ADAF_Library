"""
examples/inverse/lotka_volterra_inverse.py

Physics-Informed Inverse Training for the Lotka-Volterra predator-prey system.

True parameters (from forward example):
    alpha = 2.0   (prey birth rate)    ← estimated
    beta  = 0.04  (predation rate)     ← fixed (known)
    gamma = 1.06  (predator death)     ← estimated
    delta = 0.02  (predator growth)    ← fixed (known)

Workflow:
    1. Radau reference — generate observations from an INDEPENDENT high-accuracy
       scipy solve_ivp (Radau) integration, NOT from the ADA forward solution.
       Fitting ADA-generated data with the ADA representation is an inverse
       crime (same approximation family); a foreign reference is a fair test.
    2. sample + noise — draw noisy observations from the Radau reference
    3. run_inverse    — recover alpha and gamma from observations

"""
import os
import sys
import pathlib

# ── Path setup (editable install) ─────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent          # adalib_project/
sys.path.insert(0, str(_PROJECT_ROOT))

import adalib
from adalib import (
    run_forward,
    run_inverse,
    data_gen,
    get_system,
    InverseParameter,
    InverseOptions,
    ForwardOptions,
)

# ── System ────────────────────────────────────────────────────────────────
system = get_system("lotka_volterra")

TRUE_ALPHA = 2.0
TRUE_BETA  = 0.04
TRUE_GAMMA = 1.06
TRUE_DELTA = 0.02

x0     = [0.5, 0.075]
t_span = (0.0, 1.0)

# ── Step 1: Independent Radau reference (NOT ADA) ─────────────────────────
print("=" * 60)
print("Step 1: Independent Radau reference (avoids inverse crime)")
print("=" * 60)

import numpy as np
from scipy.integrate import solve_ivp
from adalib.inverse.observation import ObservationData

def _lv_rhs(t, y):
    prey, pred = y
    return [TRUE_ALPHA * prey - TRUE_BETA * prey * pred,
            TRUE_DELTA * prey * pred - TRUE_GAMMA * pred]

_ref = solve_ivp(_lv_rhs, t_span, x0,
                 t_eval=np.linspace(t_span[0], t_span[1], 4000),
                 method="Radau", rtol=1e-11, atol=1e-12)
print(f"  Radau reference done. success={_ref.success}")

# ── Step 2: Sample noisy observations from the reference ──────────────────
print("\nStep 2: Sample noisy observations from the Radau reference")
print("=" * 60)

_rng = np.random.default_rng(42)
_idx = np.linspace(1, _ref.t.size - 1, 200).astype(int)
t_obs = _ref.t[_idx]
y_clean = _ref.y[:, _idx].T                      # (200, 2)
_rms = np.sqrt(np.mean(_ref.y ** 2, axis=1))     # per-state scale
y_obs = y_clean + _rng.normal(0.0, 1.0, y_clean.shape) * (0.01 * _rms)[None, :]
obs = ObservationData(t=t_obs, y=y_obs, state_indices=[0, 1])
print(f"  {obs}")

# ── Step 3: Inverse training ───────────────────────────────────────────────
print("\nStep 3: Inverse training")
print("=" * 60)
print(f"  True:    alpha={TRUE_ALPHA}, gamma={TRUE_GAMMA}")
print(f"  Initial: alpha=1.5,   gamma=1.2")

inv_opts = InverseOptions(
    n_seg=10,
    N_p=5,
    N_m=100,
    Nt_total=1000,
    gamma=0.8,
    L=1.0,
    lambda_physics=1.0,
    lambda_data=10.0,
    epochs=5,
    adam_inner=200,
    adam_lr=1e-3,
    use_lbfgs=True,
    verbose=True,
    param_log_every=1,
    dtype="float64",
)

inv_result = run_inverse(
    system,
    x0=x0,
    t_span=t_span,
    params={
        "alpha": InverseParameter(initial=1.5, lower=0.0),
        "beta":  TRUE_BETA,                                   # fixed
        "gamma": InverseParameter(initial=1.2, lower=0.0),
        "delta": TRUE_DELTA,                                  # fixed
    },
    data=obs,
    options=inv_opts,
)

# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Inverse training results")
print("=" * 60)
est = inv_result.estimated_params
print(f"  alpha: true={TRUE_ALPHA:.4f}  estimated={est['alpha']:.4f}  "
      f"error={abs(est['alpha'] - TRUE_ALPHA) / TRUE_ALPHA * 100:.2f}%")
print(f"  gamma: true={TRUE_GAMMA:.4f}  estimated={est['gamma']:.4f}  "
      f"error={abs(est['gamma'] - TRUE_GAMMA) / TRUE_GAMMA * 100:.2f}%")
print(f"\n  Final loss: {inv_result.loss_history[-1]:.4e}")
print(f"  Runtime:    {inv_result.runtime_sec:.1f} s")

# ── Save figures ───────────────────────────────────────────────────────────
out_dir = _HERE / "output"
out_dir.mkdir(exist_ok=True)

fig_t, fig_p = inv_result.plot(
    state_names=["prey", "predator"],
    save_path=str(out_dir / "lv_inverse"),
    observation_data=obs,
    title="Lotka-Volterra Inverse: recovered trajectory",
    true_params={"alpha": TRUE_ALPHA, "gamma": TRUE_GAMMA},
)
fig_loss = inv_result.plot_loss(
    save_path=str(out_dir / "lv_inverse_loss.png")
)
print(f"\n  Figures saved to {out_dir}/")
