import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import adalib

adalib.utils.set_adalib_plot_style(style="serif")
plt.rcParams.update({
    "xtick.top":         False,
    "ytick.right":       False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

# ── 1. System ────────────────────────────────────────────────────────
#  Euler rigid body:
#    dω1/dt = ((I2-I3)/(I2·I3)) · ω2·ω3
#    dω2/dt = ((I3-I1)/(I1·I3)) · ω1·ω3
#    dω3/dt = ((I1-I2)/(I1·I2)) · ω1·ω2
system = adalib.get_system("euler", I1=0.2, I2=0.3, I3=0.4)

# ── 2. ForwardOptions ────────────────────────────────────────────────
options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,
    N_p=10,
    N_m=100,
    Nt_total=500,
    epochs=5,
    adam_inner=100,
    use_lbfgs=True,
    dtype="float64",
    verbose=True,
)

# ── 3. Run forward ───────────────────────────────────────────────────
X0     = [1.0, 1.0, 1.0]
T_SPAN = (0.0, 2.5)

result = adalib.run_forward(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    options=options,
)

t = result.t    # (Nt_total,)
y = result.y    # (3, Nt_total)

print("\n=== Forward result ===")
print(f"t shape  : {t.shape}")
print(f"y shape  : {y.shape}")
print(f"t range  : [{t[0]:.3f}, {t[-1]:.3f}]")

# ── 4. scipy RK45 reference ──────────────────────────────────────────
sol_ref = solve_ivp(
    lambda t, x: system.rhs(t, x),
    T_SPAN, X0,
    method="RK45", t_eval=t, rtol=1e-10, atol=1e-12,
)
ref = sol_ref.y   # (3, Nt_total)

# ── 5. L2 relative error ─────────────────────────────────────────────
state_names = ["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"]
print("\n=== L2 relative error (vs scipy RK45) ===")
for i, name in enumerate(state_names):
    err = np.linalg.norm(y[i] - ref[i]) / (np.linalg.norm(ref[i]) + 1e-12)
    print(f"  {name}: {err:.4e}")

# ── 6. Plot: all states on one axis, paper style ─────────────────────
#  colored solid = ADA,  black dashed = scipy reference
_COLORS = ["C0", "C1", "C2"]   # blue / orange / green  (matches paper)

fig, ax = plt.subplots(figsize=(5, 4))

for j in range(3):
    ax.plot(t, y[j],   color=_COLORS[j], lw=1.5)
    ax.plot(t, ref[j], "k--",            lw=1.0)

ax.set_ylabel("$\\omega_1,\\ \\omega_2,\\ \\omega_3$")
ax.set_xlabel("$t$")
ax.set_xlim(t[0], t[-1])
fig.tight_layout()
fig.savefig("euler_forward_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → euler_forward_result.png")
