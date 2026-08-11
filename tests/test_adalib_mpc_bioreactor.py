import matplotlib
matplotlib.use("Agg")
import os, sys, argparse
import numpy as np
from datetime import datetime

# ── Path & env setup ──────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "bioreactor_mpc_outputs")
_LEGACY    = os.path.abspath(os.path.join(
    _HERE, "..", "adalib", "_vendor", "legacy",
    "operator_mpc_original", "cstr_mpc_op"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("PROBLEM", "bioreactor")
os.environ.setdefault("BASIS",   "lpa")

if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

# ── Legacy imports ────────────────────────────────────────────────────────
from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0, get_data_paths
from problems.registry import get_problem
from models.learner import OperatorLearner
from data.dataset_builder import load_segments

from main_mpc_bioreactor import (
    find_latest_weight, run_mpc, run_baseline, make_plots, save_csv,
    INP_MIN, INP_MAX, YX_FIXED, SIN_FIXED, T_SEG_MIN, IC_DEFAULT,
    STATE_LABELS, STATE_COLORS, BASELINE_LS, V_MAX_DEFAULT, S_INH,
    X_MIN, X_MAX,
)

# ── 1. Load operator ──────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Load operator")
print("=" * 60)

paths     = get_data_paths("bioreactor")
train_seg = load_segments(paths["train_segment"])
x_mean    = train_seg.get("X_mean")
x_std     = train_seg.get("X_std")
print(f"  x_mean = {x_mean}")
print(f"  x_std  = {x_std}")

problem = get_problem("bioreactor")
learner = OperatorLearner(
    problem  = problem,
    hidden   = HIDDEN,
    n_layers = N_LAYERS,
    lr       = 1e-3,
    x_mean   = x_mean,
    x_std    = x_std,
)

weight_path = find_latest_weight()
learner.load_weights(weight_path)
print(f"  weights : {weight_path}")
print(f"  T_seg   : {T_SEG_MIN:.2f} min  N_SEG={N_SEG}")

# ── 2. MPC configuration ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Economic MPC")
print("=" * 60)

N_STEPS = N_SEG    # one full batch run
N_PRED  = 10       # prediction horizon [segments]
x0      = IC_DEFAULT.copy()
print(f"  n_steps={N_STEPS}  n_pred={N_PRED}")
print(f"  IC: Xs={x0[0]:.3f}  Ss={x0[1]:.3f}  Ps={x0[2]:.3f}  Vs={x0[3]:.3f} L")

args = argparse.Namespace(
    n_steps     = N_STEPS,
    n_pred      = N_PRED,
    inp_init    = 0.04,
    Yx          = YX_FIXED,
    Sin         = SIN_FIXED,
    w_product   = 1.0,
    w_stage     = 0.1,
    w_feed      = 0.01,
    w_smooth    = 0.1,
    w_volume    = 10.0,
    w_substrate = 0.5,
    w_nonneg    = 10.0,
    w_ss_track  = 0.0,
    Ss_opt      = 0.5,
    V_max       = V_MAX_DEFAULT,
    maxiter     = 150,
    ftol        = 1e-5,
)

mpc_result = run_mpc(learner, problem, x0, args)
product_final = float(mpc_result["x_arr"][-1, 2] * mpc_result["x_arr"][-1, 3])
feed_mpc      = float(np.sum(mpc_result["inp_arr"]) * T_SEG_MIN)
print(f"\n  Terminal Ps×Vs  = {product_final:.4f} g")
print(f"  Total feed      = {feed_mpc:.4f} L")
print(f"  Mean solve time = {mpc_result['t_solve'].mean():.1f} ms  "
      f"max = {mpc_result['t_solve'].max():.1f} ms")

# ── 3. Constant-feed baselines ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Baselines")
print("=" * 60)

baselines = {}
for name, inp_val in [("low", 0.005), ("nominal", 0.04), ("high", 0.20)]:
    print(f"  const-{name}  inp={inp_val:.3f} L/min", end="")
    baselines[name] = run_baseline(problem, x0, inp_val, N_STEPS)
    print(f"  →  Ps×Vs = {baselines[name]['product_final']:.4f} g")

# ── 4. Results summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
T_span = T_FINAL - T0
print(f"  {'Strategy':<20}  {'Ps×Vs [g]':>10}  {'Feed [L]':>8}  {'g/min':>9}")
print(f"  {'-'*54}")
print(f"  {'Economic MPC':<20}  {product_final:>10.4f}  "
      f"{feed_mpc:>8.4f}  {product_final/T_span:>9.5f}")
for name, br in baselines.items():
    fb = float(np.sum(br["inp_arr"]) * T_SEG_MIN)
    print(f"  {f'const {name}':<20}  {br['product_final']:>10.4f}  "
          f"{fb:>8.4f}  {br['product_final']/T_span:>9.5f}")

# ── 5. Plots & CSV ────────────────────────────────────────────────────────
t_arr = np.arange(N_STEPS + 1) * T_SEG_MIN
ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

make_plots(OUTPUT_DIR, ts, t_arr, mpc_result, baselines, args)
save_csv(OUTPUT_DIR,   ts, t_arr, mpc_result, baselines)

print(f"\nOutputs saved → {OUTPUT_DIR}/")
for fname in [f"states_{ts}.png", f"control_{ts}.png", f"solve_time_{ts}.png"]:
    print(f"  {fname}")
