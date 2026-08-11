import numpy as np
import matplotlib
matplotlib.use("Agg")
import adalib

adalib.utils.set_adalib_plot_style()

# ── 1. Define the ODE system ────────────────────────────────────────
#  Exponential decay:  dy/dt = -y   →   y(t) = exp(-t)

def rhs(t, x, u=None, p=None):
    return [-x[0]]

def rhs_tf(var_list, i, u=None, p=None):
    y, y_t = var_list[0]          # y and dy/dt at collocation point i
    return y_t - (-y)             # physics residual:  dy/dt - f(y) = 0

system = adalib.CallableODESystem(
    name="exponential_decay",
    rhs=rhs,
    rhs_tf=rhs_tf,
    state_names=["y"],
)

# ── 2. Define options ───────────────────────────────────────────────
options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,          # piecewise segments
    N_p=5,             # Legendre/Fourier modes
    N_m=50,            # Fourier truncation
    Nt_total=300,      # total output time-grid points
    epochs=5,          # Adam outer epochs
    adam_inner=100,    # Adam steps per epoch
    use_lbfgs=True,    # L-BFGS polish after Adam (requires rhs_tf)
    dtype="float64",
    verbose=True,
)

# ── 3. Run forward workflow ─────────────────────────────────────────
result = adalib.run_forward(
    system=system,
    x0=[1.0],
    t_span=(0.0, 3.0),
    options=options,
)

# ── 4. Inspect result ───────────────────────────────────────────────
t = result.t    # shape (Nt_total,)   — shortcut for result.solution.t
y = result.y    # shape (n_state, Nt_total)  — shortcut for result.solution.y

exact = np.exp(-t)

print("\n=== Forward result ===")
print(f"t shape  : {t.shape}")
print(f"y shape  : {y.shape}")
print(f"t range  : [{t[0]:.3f}, {t[-1]:.3f}]")
print(f"y(0)     = {y[0, 0]:.6f}   (exact 1.000000)")
print(f"y(3)     = {y[0, -1]:.6f}   (exact {np.exp(-3.0):.6f})")
l2_err = np.linalg.norm(y[0] - exact) / np.linalg.norm(exact)
print(f"max |err|: {np.max(np.abs(y[0] - exact)):.4e}")
print(f"L2 rel err: {l2_err:.4e}")

# ── 5. Plot vs scipy reference ──────────────────────────────────────
fig, axes = result.forward_plot(
    state_names = ["$y$"],
    save_path   = "forward_result.png",
    show        = False,
)
print("Plot saved → forward_result.png")
