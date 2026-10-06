"""
examples/simple_api/04_mpc_example.py
ADALib — Feature 3: Operator-based MPC (CSTR tracking, 5 ICs).

Demonstrates run_mpc() on the built-in CSTR system with T_R tracking.
Reuses a pre-trained operator checkpoint and runs closed-loop MPC for
5 different initial conditions.

Install once:
    pip install -e .

Run from examples/simple_api/:
    python 04_mpc_example.py
"""
import matplotlib
matplotlib.use("Agg")
import adalib

adalib.utils.set_adalib_plot_style()

# ── 1. Select a built-in system ─────────────────────────────────────
system = adalib.get_system("cstr")

# ── 2. Initial condition list ────────────────────────────────────────
IC_LIST = [
    [0.8,  0.5,  141.0, 141.0],   # T_R >> T_ref
    [1.5,  0.9,  138.5, 136.0],   # T_R slightly above
    [1.2,  0.7,  134.0, 131.0],   # T_R below T_ref
    [0.4,  0.2,  125.0, 120.0],   # T_R far below
    [1.8,  1.3,  136.5, 135.0],   # T_R ≈ T_ref (different concentrations)
]

T_REF   = 136.0
N_STEPS = 20

# ── 3. Options ───────────────────────────────────────────────────────
options_mpc = adalib.MPCOptions(
    mode="tracking",
    basis="lpa",

    target={"T_R": T_REF},
    n_steps=N_STEPS,

    # Data/training already done → reuse
    generate_data=False,
    reuse_existing_data=True,
    train_operator=False,
    reuse_existing_operator=True,
    epochs=1000,
    batch_size=8,
    lr=3e-3,
    hidden=64,
    n_layers=2,

    run_closed_loop=True,
    work_dir="./runs/simple_mpc_cstr",
    verbose=False,
)

# ── 4. Run MPC for 5 ICs ─────────────────────────────────────────────
print("Running MPC for 5 ICs ...")
all_results = []
for ic_idx, x0 in enumerate(IC_LIST):
    print(f"  IC {ic_idx+1}: {x0}")
    r = adalib.run_mpc(
        system=system,
        x0=x0,
        t_span=(0.0, 0.5),
        options=options_mpc,
    )
    all_results.append(r)
    print(f"    → T_R final: {r.x[-1, 2]:.2f} °C  (target {T_REF} °C)")

# ── 5. Plot MPC trajectory ───────────────────────────────────────────
state_names   = ["$C_A$ [mol/l]", "$C_B$ [mol/l]", "$T_R$ [°C]", "$T_K$ [°C]"]
control_names = ["$\\dot{Q}$ [kJ/h]"]

col_labels = [
    f"IC {i+1}\n"
    f"$C_A$={IC_LIST[i][0]:.2f}, $C_B$={IC_LIST[i][1]:.2f}\n"
    f"$T_R$={IC_LIST[i][2]:.1f}°C, $T_K$={IC_LIST[i][3]:.1f}°C"
    for i in range(len(IC_LIST))
]

# Single-IC closed-loop result
fig, axes = all_results[0].MPC_result(
    state_names   = state_names,
    control_names = control_names,
    target        = {"T_R": T_REF},
    title         = f"CSTR MPC — IC 1  (n_steps={N_STEPS}, $T_{{ref}}$={T_REF}°C)",
    save_path     = "mpc_result_ic1.png",
    show          = False,
)
print("\nSingle-IC plot saved → mpc_result_ic1.png")

# Full 5-IC comparison
fig2, axes2 = adalib.utils.plot_mpc_result(
    all_results,
    state_names   = state_names,
    control_names = control_names,
    target        = {"T_R": T_REF},
    labels        = col_labels,
    title         = f"CSTR MPC — {len(IC_LIST)} Initial Conditions  "
                    f"(n_steps={N_STEPS}, $T_{{ref}}$={T_REF}°C)",
    save_path     = "mpc_result.png",
    show          = False,
)
print("5-IC plot saved → mpc_result.png")

# ── 6. Validate operator surrogate ──────────────────────────────────
fig3, axes3 = all_results[0].MPC_infer(
    n_cases     = 4,
    state_names = state_names,
    title       = "CSTR MPC — Operator surrogate validation",
    save_path   = "mpc_infer.png",
    show        = False,
)
print("Operator inference plot saved → mpc_infer.png")
