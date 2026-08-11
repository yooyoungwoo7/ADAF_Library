"""
examples/simple_api/05_generic_tracking_mpc.py
Generic Tracking MPC — Mass-Spring-Damper example

Demonstrates adalib.run_mpc() with a user-defined CallableODESystem.
The surrogate model is an ADA LPA Operator NN (same OperatorNet /
BatchLPABasis architecture as the built-in CSTR / triple-tank workflows).

System
------
  m·ẍ + c·ẋ + k·x = F
  m = 1.0 kg,  c = 0.3 N·s/m,  k = 1.5 N/m

States:  [x (position, m),  v (velocity, m/s)]
Control: [F (force, N)]
Task:    drive x → 1.0 m from x0 = 0.0 m

Run
---
  python examples/simple_api/05_generic_tracking_mpc.py
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import adalib

# ── 1. Define the ODE system ─────────────────────────────────────────
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

# ── 2. MPC options ───────────────────────────────────────────────────
opts = adalib.MPCOptions(
    mode="tracking",

    # --- generic MPC fields ---
    control_inputs=["F"],
    controlled_variables=["x"],
    target={"x": 1.0},
    dt=0.4,                        # segment duration [s]
    horizon=5,                     # prediction horizon (steps)
    tracking_weights=[10.0],       # Q: penalise position error
    control_weights=[0.05],        # R: penalise force

    # --- data & training ---
    n_train=2000,
    n_val=200,
    seed=0,
    generate_data=True,
    train_operator=True,
    epochs=2000,
    batch_size=64,
    lr=1e-3,
    hidden=64,
    n_layers=3,

    # --- closed-loop ---
    n_steps=25,
    run_closed_loop=True,
    work_dir="./runs/msd_generic_mpc",
    verbose=True,
)

# ── 3. Run ───────────────────────────────────────────────────────────
x0 = [0.0, 0.0]    # start at rest at origin
result = adalib.run_mpc(
    system=system,
    x0=x0,
    t_span=(0.0, opts.n_steps * opts.dt),
    options=opts,
)

# ── 4. Inspect ───────────────────────────────────────────────────────
print("\n=== MPCResult ===")
print(f"t shape : {result.t.shape}")
print(f"x shape : {result.x.shape}")
print(f"u shape : {result.u.shape}")
print(f"x_final : x={result.x[-1, 0]:.4f}  v={result.x[-1, 1]:.4f}")
print(f"target  : x=1.0000")
print(f"\nPaths:")
for k, v in result.paths.items():
    if v:
        print(f"  {k:<20}: {v}")

# ── 5. Plot via result.plot() ────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = result.MPC_result(
        state_names   = ["$x$ [m]", "$v$ [m/s]"],
        control_names = ["$F$ [N]"],
        target        = {"x": 1.0},
        title         = "Mass-Spring-Damper — Generic Tracking MPC",
        save_path     = "msd_generic_mpc.png",
        show          = False,
    )
    print("\nPlot saved → msd_generic_mpc.png")

    # Artifact list
    arts = result.list_artifacts()
    print("\n=== Artifacts ===")
    for a in arts:
        print(f"  {a}")
except Exception as e:
    print(f"Plot skipped: {e}")
