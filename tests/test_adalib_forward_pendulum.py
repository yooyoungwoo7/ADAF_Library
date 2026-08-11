import numpy as np
import matplotlib
matplotlib.use("Agg")
import adalib

from utils._style import apply_style
apply_style()

# ── 1. System: Damped Nonlinear Pendulum ─────────────────────────────
#
#  d²θ/dt² + γ · dθ/dt + (g/L) · sin(θ) = 0
#
#  States:  θ [rad]   — angular displacement
#           ω [rad/s] — angular velocity (= dθ/dt)
#
#  Parameter values
#    γ   = 0.30 /s   (viscous damping at pivot)
#    g/L = 9.81 rad/s²  (L = 1 m rod length)

GAMMA    = 0.30
G_OVER_L = 9.81

def pendulum_rhs(t, state, u=None, p=None):
    theta, omega = state
    return [omega,
            -GAMMA * omega - G_OVER_L * np.sin(theta)]

def pendulum_rhs_tf(var_list, i, u=None, p=None):
    import tensorflow as tf
    theta, theta_t = var_list[0]
    omega, omega_t = var_list[1]
    if i == 0:
        return theta_t - omega
    else:
        return omega_t - (-GAMMA * omega - G_OVER_L * tf.sin(theta))

system = adalib.CallableODESystem(
    name="damped_pendulum",
    rhs=pendulum_rhs,
    rhs_tf=pendulum_rhs_tf,
    state_names=["theta", "omega"],
)

# ── 2. ForwardOptions ────────────────────────────────────────────────
options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,
    N_p=10,
    N_m=100,
    Nt_total=1000,
    epochs=5,
    adam_inner=100,
    use_lbfgs=True,
    dtype="float64",
    verbose=True,
)

# ── 3. Run forward ───────────────────────────────────────────────────
result = adalib.run_forward(
    system=system,
    x0=[np.pi / 3, 0.0],   # 60° release from rest
    t_span=(0.0, 10.0),
    options=options,
)

# ── 4. Inspect result ────────────────────────────────────────────────
print("\n=== Damped Pendulum — Forward Result ===")
print(f"t:  {result.t.shape}   range [{result.t[0]:.2f}, {result.t[-1]:.2f}] s")
print(f"θ:  [{result.y[0].min():.4f}, {result.y[0].max():.4f}] rad")
print(f"ω:  [{result.y[1].min():.4f}, {result.y[1].max():.4f}] rad/s")

# ── 5. Plot vs scipy reference ───────────────────────────────────────
fig, axes = result.forward_plot(
    state_names = [r"$\theta$ [rad]", r"$\omega$ [rad/s]"],
    save_path   = "pendulum_forward_result.png",
    show        = False,
)
print("\nPlot saved → pendulum_forward_result.png")
