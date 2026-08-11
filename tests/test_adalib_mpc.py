import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import adalib

adalib.utils.set_adalib_plot_style("serif")

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
    f"$C_A$={IC_LIST[i][0]:.2f}, $C_B$={IC_LIST[i][1]:.2f}\n"
    f"$T_R$={IC_LIST[i][2]:.1f}°C, $T_K$={IC_LIST[i][3]:.1f}°C"
    for i in range(len(IC_LIST))
]

# Single-IC closed-loop result
fig0, axes0 = all_results[0].MPC_result(
    state_names   = state_names,
    control_names = control_names,
    target        = {"T_R": T_REF},
    title         = f"CSTR MPC — IC 1  (n_steps={N_STEPS}, $T_{{ref}}$={T_REF}°C)",
    save_path     = "mpc_result_ic1.png",
    show          = False,
)
print("\nSingle-IC plot saved → mpc_result_ic1.png")

# Cases 1, 4, 5 (0-indexed: 0, 3, 4)  — remove similar mid-range cases
PLOT_ICS = [0, 3, 4]
plot_results = [all_results[i] for i in PLOT_ICS]
plot_labels  = [col_labels[i]  for i in PLOT_ICS]

fig, axes = adalib.utils.plot_mpc_result(
    plot_results,
    state_names   = state_names,
    control_names = control_names,
    target        = {"T_R": T_REF},
    labels        = plot_labels,
    state_groups  = [[0, 1], [2, 3]],   # row 0: C_A & C_B,  row 1: T_R & T_K
    title         = "",                  # suppress suptitle
    save_path     = "mpc_result.png",
    show          = False,
)
print("3-IC plot saved → mpc_result.png")

# ── 6. Inference plot: operator surrogate vs BDF plant (ground truth) ────
#
#  After run_mpc, legacy modules are in sys.modules.
#  res.x = TRUE plant trajectory (BDF) → this is the "reference"
#  We do a fresh NN operator rollout with the same (x0, u) sequence.
import glob as _glob
import config as _cfg
import problems.registry as _preg
import models.learner as _ml
from adalib.workflows.mpc_workflow import _predict_next_scalar

_problem = _preg.get_problem("cstr_mpc")

# Load normalization stats from training segments
_data_dir = all_results[0].paths["data_dir"]
_seg      = np.load(os.path.join(_data_dir, "cstr_mpc_train_segments.npz"),
                    allow_pickle=True)

# Instantiate & reload the operator learner
_learner = _ml.OperatorLearner(
    problem  = _problem,
    hidden   = options_mpc.hidden,
    n_layers = options_mpc.n_layers,
    lr       = options_mpc.lr,
    x_mean   = _seg["X_mean"],
    x_std    = _seg["X_std"],
)
_ckpt_dir   = all_results[0].paths["checkpoint_dir"]
_ckpt_files = sorted(_glob.glob(os.path.join(_ckpt_dir, "*best*.weights.h5")))
_learner.load_weights(_ckpt_files[-1])
print(f"[Inference] Loaded operator weights: {os.path.basename(_ckpt_files[-1])}")

_X_MIN = np.array([0.0, 0.0,  50.0,  50.0], dtype=np.float32)
_X_MAX = np.array([5.0, 5.0, 200.0, 200.0], dtype=np.float32)

def get_operator_rollout(res):
    """Fresh NN operator rollout with same x0 and MPC control sequence."""
    xk    = np.asarray(res.x[0], dtype=np.float32)
    u_seq = np.asarray(res.u, dtype=float).ravel()
    x_op  = [xk.astype(float)]
    for Q_k in u_seq:
        xk = _predict_next_scalar(_learner, xk, float(Q_k), _X_MIN, _X_MAX)
        x_op.append(xk.astype(float))
    return np.stack(x_op)   # (n_steps+1, n_state)

print("\nComputing operator rollouts ...")
from dataclasses import replace as _dc_replace

op_results  = []   # MPCResult with x = operator rollout
references  = []   # plant (BDF) trajectories as reference
infer_labels = []

for k, i_ic in enumerate(PLOT_ICS):
    res   = all_results[i_ic]
    x_op  = get_operator_rollout(res)
    x_ref = np.asarray(res.x, dtype=float)   # BDF plant = ground truth

    # L2 error: operator vs plant
    l2s = [np.linalg.norm(x_op[:, s] - x_ref[:, s]) /
           (np.linalg.norm(x_ref[:, s]) + 1e-12) for s in range(4)]
    mean_l2 = float(np.mean(l2s))
    print(f"  Case {i_ic+1}: mean L2 = {mean_l2:.2e}")

    op_results.append(_dc_replace(res, x=x_op.astype(np.float32)))
    references.append(x_ref)
    infer_labels.append(f"{plot_labels[k]}\nL2={mean_l2:.2e}")

fig_inf, axes_inf = adalib.utils.plot_mpc_result(
    op_results,
    state_names     = state_names,
    control_names   = control_names,
    target          = {"T_R": T_REF},
    labels          = infer_labels,
    state_groups    = [[0, 1], [2, 3]],
    references      = references,        # plant/BDF = black dashed reference
    title           = "",
    save_path       = "mpc_inference.png",
    show            = False,
)
print("Inference plot saved → mpc_inference.png")

# ── 7. Validate operator surrogate (legacy) ──────────────────────────
fig2, axes2 = all_results[0].MPC_infer(
    n_cases     = 4,
    state_names = state_names,
    title       = "CSTR MPC — Operator surrogate validation",
    save_path   = "mpc_infer.png",
    show        = False,
)
print("Operator inference plot saved → mpc_infer.png")
