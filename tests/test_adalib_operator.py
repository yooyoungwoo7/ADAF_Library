import numpy as np
import matplotlib
matplotlib.use("Agg")
import adalib

adalib.utils.set_adalib_plot_style(style="serif")

# ── 1. Select a built-in system ─────────────────────────────────────
system = adalib.get_system("cstr")

# ── 2. Define options ───────────────────────────────────────────────
options = adalib.OperatorOptions(
    basis="lpa",

    # Data generation
    n_train=2000,
    n_val=200,
    seed=42,
    generate_data=False,
    reuse_existing_data=True,

    # Training
    train=False,
    reuse_existing_checkpoint=True,
    epochs=1000,
    batch_size=8,
    lr=3e-3,
    hidden=64,
    n_layers=2,

    # Inference after training
    infer=True,

    # All artifacts (data, checkpoints, logs, results) go here
    work_dir="./runs/simple_operator_cstr",

    verbose=True,
)

# ── 3. Run operator workflow ────────────────────────────────────────
result = adalib.run_operator(
    system=system,
    x0=[0.8, 0.5, 134.14, 130.0],   # [C_A, C_B, T_R, T_K]
    t_span=(0.0, 0.5),               # informational; segment grid from config
    params=[1.0, 1.0, 50.0, -2000.0],
    options=options,
)

# ── 4. Inspect result ───────────────────────────────────────────────
print("\n=== OperatorResult ===")
print(f"t shape  : {result.t.shape}")    # (N_SEG + 1,)
print(f"y shape  : {result.y.shape}")    # (N_SEG + 1, n_state)
print(f"t range  : [{result.t[0]:.3f}, {result.t[-1]:.3f}] h")
print(f"y(t=0)   : {result.y[0]}")
print(f"y(t_end) : {result.y[-1]}")

print("\n=== Artifact paths ===")
print(f"work_dir       : {result.paths['work_dir']}")
print(f"data_dir       : {result.paths['data_dir']}")
print(f"checkpoint_dir : {result.paths['checkpoint_dir']}")
print(f"result_dir     : {result.paths['result_dir']}")

print("\n=== Metadata ===")
for k, v in result.metadata.items():
    print(f"  {k:<22}: {v}")

# ── 5. Plot rollout vs scipy BDF reference (Case 1) ─────────────────
state_names  = ["C_A",          "C_B",          "T_R",       "T_K"]
state_labels = ["$C_A$ [mol/l]","$C_B$ [mol/l]","$T_R$ [°C]","$T_K$ [°C]"]

fig, axes, metrics = result.plot(
    reference   = "solve_ivp",
    params      = [1.0, 1.0, 50.0, -2000.0],
    state_names = state_labels,
    state_groups= [[0, 1], [2, 3]],
    title       = "CSTR — operator vs scipy BDF reference (25 segments)",
    save_path   = "operator_result.png",
    show        = False,
)
print("\nPlot saved → operator_result.png")
print("L2 rel errors (Case 1):", ", ".join(
    f"{n}={v:.2e}" for n, v in zip(state_names, metrics["l2_rel"][0])))

# ── 6. Multi-case inference validation ──────────────────────────────
fig2, axes2 = result.operator_infer(
    n_cases     = 4,
    state_names = state_labels,
    title       = "CSTR — LPA operator inference (4 test cases)",
    save_path   = "operator_inference.png",
    show        = False,
)
print("Inference plot saved → operator_inference.png")
