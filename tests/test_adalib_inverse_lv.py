import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from scipy.integrate import solve_ivp
import adalib
from adalib.inverse.observation import ObservationData

# 출력 폴더는 이 파일 옆에 고정 (어디서 실행해도 같은 위치)
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "lv_inverse_outputs")

# ── 1. System ─────────────────────────────────────────────────────────────
#  Lotka-Volterra (normalized scale: U_scale=200, R_scale=20)
#    dx/dt =  alpha*x - beta*x*y
#    dy/dt = -gamma*y + delta*x*y
#
#  True params (scaled):
#    alpha = 2.0  * R_SCALE = 40.0   ← estimated
#    beta  = 0.04 * R_SCALE * U_SCALE = 160.0  ← fixed
#    gamma = 1.06 * R_SCALE = 21.2   ← estimated
#    delta = 0.02 * R_SCALE * U_SCALE = 80.0   ← fixed
#
#  Equilibrium: x* = gamma/delta = 0.265,  y* = alpha/beta = 0.25
#  Period     : 2π/√(alpha*gamma) ≈ 0.22  → ~4.5 cycles in [0, 1]
#
#  Why alpha/gamma (not U/R):
#    ∂L_phys/∂alpha ∝ (alpha_true - alpha) · mean(r²)  → consistent direction
#    ∂L_phys/∂gamma ∝ (gamma_true - gamma) · mean(p²)  → consistent direction
#    U/R gradient cancels over oscillation period → poor convergence

U_SCALE = 200.0
R_SCALE =  20.0

TRUE_ALPHA = 2.0  * R_SCALE            # 40.0
TRUE_BETA  = 0.04 * R_SCALE * U_SCALE  # 160.0
TRUE_GAMMA = 1.06 * R_SCALE            # 21.2
TRUE_DELTA = 0.02 * R_SCALE * U_SCALE  # 80.0

system = adalib.get_system("lotka_volterra",
    alpha=TRUE_ALPHA,
    beta=TRUE_BETA,
    gamma=TRUE_GAMMA,
    delta=TRUE_DELTA,
)

X0     = [100.0 / U_SCALE, 15.0 / U_SCALE]  # [0.5, 0.075]
T_SPAN = (0.0, 1.0)

# ── 2. Independent Radau reference (NOT ADA) — avoids the inverse crime ───
print("=" * 60)
print("Step 1: Independent Radau reference (avoids inverse crime)")
print("=" * 60)

_ref = solve_ivp(
    lambda t, x: system.rhs(t, x, p=[TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA, TRUE_DELTA]),
    T_SPAN, X0, t_eval=np.linspace(T_SPAN[0], T_SPAN[1], 4000),
    method="Radau", rtol=1e-11, atol=1e-12,
)
print(f"  Radau reference done. success={_ref.success}")

# ── 3. Sample observations from the independent reference ─────────────────
print("\nStep 2: Sample observations from the Radau reference")
print("=" * 60)

_idx = np.linspace(1, _ref.t.size - 1, 500).astype(int)
t_obs = _ref.t[_idx]
y_obs = _ref.y[:, _idx].T  # noise_std=0.0, so no noise added (matches prior setting)
obs = ObservationData(t=t_obs, y=y_obs, state_indices=[0, 1])
print(f"  {obs}")

# ── 4. Inverse training ───────────────────────────────────────────────────
print("\nStep 3: run_inverse")
print("=" * 60)
print(f"  True:    alpha={TRUE_ALPHA}, gamma={TRUE_GAMMA}")
print(f"  Initial: alpha=30.0,  gamma=15.0")

inv_options = adalib.InverseOptions(
    n_seg=30,
    N_p=20,
    N_m=100,
    Nt_total=2000,
    lambda_physics=1e0,
    lambda_data=1e0,   # key: strong data pull on W → clean phys gradient for θ
    epochs=50,
    adam_inner=100,
    adam_lr=1e-3,
    use_lbfgs=True,
    n_passes=1,
    dtype="float64",
    verbose=True,
    param_log_every=10,
    output_dir=OUTPUT_DIR,
    true_params={"alpha": TRUE_ALPHA, "gamma": TRUE_GAMMA},

)

inv_result = adalib.run_inverse(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    params={
        "alpha": adalib.InverseParameter(initial=30.0, lower=0.1),
        "beta":  TRUE_BETA,
        "gamma": adalib.InverseParameter(initial=15.0, lower=0.1),
        "delta": TRUE_DELTA,
    },
    data=obs,
    options=inv_options,
)

# ── 5. Results ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
est = inv_result.estimated_params
print(f"  alpha : true={TRUE_ALPHA:.4f}  estimated={est['alpha']:.4f}  "
      f"error={abs(est['alpha'] - TRUE_ALPHA):.4f} ({abs(est['alpha'] - TRUE_ALPHA) / TRUE_ALPHA * 100:.2f}%)")
print(f"  gamma : true={TRUE_GAMMA:.4f}  estimated={est['gamma']:.4f}  "
      f"error={abs(est['gamma'] - TRUE_GAMMA):.4f} ({abs(est['gamma'] - TRUE_GAMMA) / TRUE_GAMMA * 100:.2f}%)")
print(f"\n  Final loss : {inv_result.loss_history[-1]:.4e}")
print(f"  Runtime    : {inv_result.runtime_sec:.1f} s")

# ── 6. Plot ───────────────────────────────────────────────────────────────
fig_t, _ = inv_result.plot(
    state_names=["prey", "predator"],
    save_path="lv_inverse_result",
    observation_data=obs,
    title="Lotka-Volterra Inverse — recovered trajectory",
    true_params={"alpha": TRUE_ALPHA, "gamma": TRUE_GAMMA},
)
fig_p = inv_result.plot_params(
    true_params={"alpha": TRUE_ALPHA, "gamma": TRUE_GAMMA},
    save_path=os.path.join(OUTPUT_DIR, "lv_inverse_result_params.png"),
    figsize=(5, 4),
)
print("\nPlot saved → lv_inverse_result_trajectory.png")
print("           → lv_inverse_result_params.png")

fig_loss = inv_result.plot_loss(save_path="lv_inverse_loss.png")
print("           → lv_inverse_loss.png")
