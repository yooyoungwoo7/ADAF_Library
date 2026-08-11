import matplotlib
matplotlib.use("Agg")
import adalib

# ── 1. System ─────────────────────────────────────────────────────────────
#  Euler rigid body (Euler's rotation equations)
#    dω1/dt = ((I2-I3)/(I2·I3)) · ω2·ω3
#    dω2/dt = ((I3-I1)/(I1·I3)) · ω1·ω3
#    dω3/dt = ((I1-I2)/(I1·I2)) · ω1·ω2
#
#  Identifiability note:
#    Only the RATIOS of inertias affect the dynamics, so all three
#    cannot be estimated simultaneously.  Fix I1 and estimate I2, I3.
#
#  True params:  I1=0.2 (fixed), I2=0.3, I3=0.4
#  Unknown:      I2, I3

TRUE_I1 = 0.2
TRUE_I2 = 0.3
TRUE_I3 = 0.4

system = adalib.get_system("euler", I1=TRUE_I1, I2=TRUE_I2, I3=TRUE_I3)

X0     = [1.0, 1.0, 1.0]
T_SPAN = (0.0, 2.5)

# ── 2. Forward solve (generate ground truth) ──────────────────────────────
print("=" * 60)
print("Step 1: Forward solve")
print("=" * 60)

fwd_options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,
    N_p=10,
    N_m=100,
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
    params=[TRUE_I1, TRUE_I2, TRUE_I3],
    options=fwd_options,
)

print(f"  t shape : {fwd_result.t.shape}")
print(f"  y shape : {fwd_result.y.shape}")

# ── 3. Generate observations ──────────────────────────────────────────────
print("\nStep 2: data_gen")
print("=" * 60)

obs = adalib.data_gen(
    fwd_result,
    n_points=200,
    noise_std=0,
    seed=123,
    state_indices=[0, 1, 2],
)
print(f"  {obs}")

# ── 4. Inverse training ───────────────────────────────────────────────────
inv_options = adalib.InverseOptions(
    n_seg=10,
    N_p=5,
    N_m=100,
    Nt_total=2000,
    lambda_physics=1.0,
    lambda_data=10.0,
    epochs=5,
    adam_inner=200,
    adam_lr=1e-3,
    use_lbfgs=True,
    n_passes=1,
    dtype="float64",
    verbose=True,
    param_log_every=1,
)

inv_result = adalib.run_inverse(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    params={
        "I1": TRUE_I1,
        "I2": adalib.InverseParameter(initial=0.25, lower=0.05, upper=2.0),
        "I3": adalib.InverseParameter(initial=0.35, lower=0.05, upper=2.0),
    },
    data=obs,
    options=inv_options,
)

# ── 5. Results ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
est = inv_result.estimated_params
print(f"  I2 : true={TRUE_I2:.4f}  estimated={est['I2']:.4f}  "
      f"error={abs(est['I2'] - TRUE_I2):.4f} ({abs(est['I2'] - TRUE_I2) / TRUE_I2 * 100:.2f}%)")
print(f"  I3 : true={TRUE_I3:.4f}  estimated={est['I3']:.4f}  "
      f"error={abs(est['I3'] - TRUE_I3):.4f} ({abs(est['I3'] - TRUE_I3) / TRUE_I3 * 100:.2f}%)")
print(f"\n  Final loss : {inv_result.loss_history[-1]:.4e}")
print(f"  Runtime    : {inv_result.runtime_sec:.1f} s")
print(f"\n  t shape : {inv_result.t.shape}")
print(f"  y shape : {inv_result.y.shape}")

# ── 6. Plot ───────────────────────────────────────────────────────────────
fig_t, _ = inv_result.plot(
    state_names=["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"],
    save_path="euler_inverse_result",
    observation_data=obs,
    title="Euler Rigid Body Inverse — recovered trajectory",
    true_params={"I2": TRUE_I2, "I3": TRUE_I3},
)
fig_p = inv_result.plot_params(
    true_params={"I2": TRUE_I2, "I3": TRUE_I3},
    save_path="euler_inverse_result_params.png",
    figsize=(5, 4),
)
print("\nPlot saved → euler_inverse_result_trajectory.png")
print("           → euler_inverse_result_params.png")

fig_loss = inv_result.plot_loss(save_path="euler_inverse_loss.png")
print("           → euler_inverse_loss.png")
