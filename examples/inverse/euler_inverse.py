"""
examples/inverse/euler_inverse.py

Physics-Informed Inverse Training for the Euler Rigid Body system.

True parameters (from forward example):
    I1 = 0.2  (principal moment, x-axis)  ← fixed (known, identifiability anchor)
    I2 = 0.3  (principal moment, y-axis)  ← estimated
    I3 = 0.4  (principal moment, z-axis)  ← estimated

Note on identifiability:
    Estimating all three inertias simultaneously is an under-determined problem
    (only ratios matter for the dynamics).  We fix I1=0.2 and estimate I2, I3.

Workflow:
    1. run_forward  — solve with true inertias to generate reference
    2. data_gen     — sample 200 noisy observations of all 3 angular velocities
    3. run_inverse  — recover I2 and I3 from observations
"""
import os
import sys
import pathlib

# ── Path setup (editable install) ─────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
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
TRUE_I1 = 0.2
TRUE_I2 = 0.3
TRUE_I3 = 0.4

system = get_system("euler", I1=TRUE_I1, I2=TRUE_I2, I3=TRUE_I3)

x0     = [1.0, 1.0, 1.0]
t_span = (0.0, 2.5)

# ── Step 1: Forward solve with true inertias ──────────────────────────────
print("=" * 60)
print("Step 1: Forward solve (ground truth)")
print("=" * 60)

fwd_opts = ForwardOptions(
    n_seg=20, N_p=10, N_m=100, Nt_total=2000,
    epochs=10, adam_inner=100, adam_lr=1e-3,
    use_lbfgs=True, verbose=False,
)
fwd_result = run_forward(
    system, x0=x0, t_span=t_span,
    params=[TRUE_I1, TRUE_I2, TRUE_I3],
    options=fwd_opts,
)
print(f"  Forward done. t: {fwd_result.t[0]:.3f} → {fwd_result.t[-1]:.3f}")

# ── Step 2: Generate observations ─────────────────────────────────────────
print("\nStep 2: Generate synthetic observations")
print("=" * 60)

obs = data_gen(
    fwd_result,
    n_points=200,
    noise_std=0.01,
    seed=123,
    state_indices=[0, 1, 2],   # observe all three angular velocities
)
print(f"  {obs}")

# ── Step 3: Inverse training ───────────────────────────────────────────────
print("\nStep 3: Inverse training")
print("=" * 60)
print(f"  True:    I2={TRUE_I2}, I3={TRUE_I3}")
print(f"  Initial: I2=0.25,     I3=0.35")
print(f"  Fixed:   I1={TRUE_I1}  (identifiability anchor)")

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
        "I1": TRUE_I1,                                       # fixed
        "I2": InverseParameter(initial=0.25, lower=0.05, upper=2.0),
        "I3": InverseParameter(initial=0.35, lower=0.05, upper=2.0),
    },
    data=obs,
    options=inv_opts,
)

# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Inverse training results")
print("=" * 60)
est = inv_result.estimated_params
print(f"  I2: true={TRUE_I2:.4f}  estimated={est['I2']:.4f}  "
      f"error={abs(est['I2'] - TRUE_I2) / TRUE_I2 * 100:.2f}%")
print(f"  I3: true={TRUE_I3:.4f}  estimated={est['I3']:.4f}  "
      f"error={abs(est['I3'] - TRUE_I3) / TRUE_I3 * 100:.2f}%")
print(f"\n  Final loss: {inv_result.loss_history[-1]:.4e}")
print(f"  Runtime:    {inv_result.runtime_sec:.1f} s")

# ── Save figures ───────────────────────────────────────────────────────────
out_dir = _HERE / "output"
out_dir.mkdir(exist_ok=True)

fig_t, fig_p = inv_result.plot(
    state_names=["omega1", "omega2", "omega3"],
    save_path=str(out_dir / "euler_inverse"),
    observation_data=obs,
    title="Euler Rigid Body Inverse: recovered trajectory",
    true_params={"I2": TRUE_I2, "I3": TRUE_I3},
)
fig_loss = inv_result.plot_loss(
    save_path=str(out_dir / "euler_inverse_loss.png")
)
print(f"\n  Figures saved to {out_dir}/")
