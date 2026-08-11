import matplotlib
matplotlib.use("Agg")
import adalib

# ── 1. System: Damped Nonlinear Pendulum ──────────────────────────────────
#
#  d(theta)/dt = omega
#  d(omega)/dt = -gamma * omega - g_over_l * sin(theta)
#
#  True params:
#    gamma    = 0.30 [1/s]   — viscous damping         ← estimated
#    g_over_l = 9.81 [rad/s²] — g/L, L=1 m rod length  ← estimated
#
#  IC: theta0 = pi/3 (60 deg), omega0 = 0  → ~5 damped cycles in [0, 10]
#  Period: 2*pi / sqrt(9.81) ≈ 2.01 s  →  ~5 cycles in 10 s
#
#  Gradient analysis (why this works better than LV U/R):
#    dL/d(gamma)    ∝ mean(res_1 · omega)        — omega decays but stays signed
#    dL/d(g_over_l) ∝ mean(res_1 · sin(theta))   — sin(theta) oscillates but
#                                                    decaying amplitude → less cancellation
#    Both gradients have consistent net direction unlike LV's r·p over full cycles.

TRUE_GAMMA    = 0.30
TRUE_G_OVER_L = 9.81

system = adalib.get_system("damped_pendulum",
    gamma=TRUE_GAMMA, g_over_l=TRUE_G_OVER_L)

X0     = [3.14159265358979 / 3, 0.0]  # [pi/3, 0]
T_SPAN = (0.0, 10.0)

# ── 2. Forward solve (generate ground truth) ──────────────────────────────
print("=" * 60)
print("Step 1: Forward solve")
print("=" * 60)

fwd_options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,
    N_p=5,
    N_m=50,
    Nt_total=2000,
    epochs=5,
    adam_inner=100,
    use_lbfgs=True,
    dtype="float64",
    verbose=False,
)

fwd_result = adalib.run_forward(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    params=[TRUE_GAMMA, TRUE_G_OVER_L],
    options=fwd_options,
)

print(f"  t shape : {fwd_result.t.shape}")
print(f"  theta   : [{fwd_result.y[0].min():.4f}, {fwd_result.y[0].max():.4f}] rad")
print(f"  omega   : [{fwd_result.y[1].min():.4f}, {fwd_result.y[1].max():.4f}] rad/s")

# ── 3. Generate observations ──────────────────────────────────────────────
print("\nStep 2: data_gen")
print("=" * 60)

obs = adalib.data_gen(
    fwd_result,
    n_points=500,
    noise_std=0.0,
    seed=42,
    state_indices=[0, 1],
)
print(f"  {obs}")

# ── 4. Inverse training ───────────────────────────────────────────────────
print("\nStep 3: run_inverse")
print("=" * 60)
print(f"  True:    gamma={TRUE_GAMMA}, g_over_l={TRUE_G_OVER_L}")
print(f"  Initial: gamma=0.10,  g_over_l=5.0")

inv_options = adalib.InverseOptions(
    n_seg=20,
    N_p=5,
    N_m=20,
    Nt_total=4000,          # Nt_seg = 200 >> N_m = 20
    lambda_physics=1.0,
    lambda_data=500.0,      # strong data pull → clean physics gradient for params
    epochs=3,
    adam_inner=100,
    adam_lr=1e-3,
    use_lbfgs=True,
    n_passes=1,
    dtype="float64",
    verbose=True,
    param_log_every=1,
    training_strategy="alternating",  # designed for W→fit data, θ→physics only
)

inv_result = adalib.run_inverse(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    params={
        "gamma":    adalib.InverseParameter(initial=0.10, lower=1e-4),
        "g_over_l": adalib.InverseParameter(initial=5.0,  lower=0.1),
    },
    data=obs,
    options=inv_options,
)

# ── 5. Results ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
est = inv_result.estimated_params

gamma_err    = abs(est["gamma"]    - TRUE_GAMMA)    / TRUE_GAMMA    * 100
g_over_l_err = abs(est["g_over_l"] - TRUE_G_OVER_L) / TRUE_G_OVER_L * 100

print(f"  gamma    : true={TRUE_GAMMA:.4f}   estimated={est['gamma']:.4f}   "
      f"error={abs(est['gamma'] - TRUE_GAMMA):.4f} ({gamma_err:.2f}%)")
print(f"  g_over_l : true={TRUE_G_OVER_L:.4f}  estimated={est['g_over_l']:.4f}  "
      f"error={abs(est['g_over_l'] - TRUE_G_OVER_L):.4f} ({g_over_l_err:.2f}%)")
print(f"\n  Final loss : {inv_result.loss_history[-1]:.4e}")
print(f"  Runtime    : {inv_result.runtime_sec:.1f} s")

# ── 6. Plot ───────────────────────────────────────────────────────────────
fig_t, fig_p = inv_result.plot(
    state_names=["theta [rad]", "omega [rad/s]"],
    save_path="pendulum_inverse_result",
    observation_data=obs,
    title="Damped Pendulum Inverse — recovered trajectory",
    true_params={"gamma": TRUE_GAMMA, "g_over_l": TRUE_G_OVER_L},
)
print("\nPlot saved -> pendulum_inverse_result_trajectory.png")
print("           -> pendulum_inverse_result_params.png")

fig_loss = inv_result.plot_loss(save_path="pendulum_inverse_loss.png")
print("           -> pendulum_inverse_loss.png")
