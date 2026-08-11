import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import adalib

# ── Paper style: serif, closed-box, inward ticks ─────────────────────
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
#  Lotka-Volterra (normalized scale: U_scale=200, R_scale=20)
U_SCALE = 200.0
R_SCALE =  20.0

system = adalib.get_system("lotka_volterra",
    alpha = 2.0  * R_SCALE,            # 40.0
    beta  = 0.04 * R_SCALE * U_SCALE,  # 160.0
    gamma = 1.06 * R_SCALE,            # 21.2
    delta = 0.02 * R_SCALE * U_SCALE,  # 80.0
)

# ── 2. ForwardOptions ────────────────────────────────────────────────
options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=50,
    N_p=5,
    N_m=100,
    Nt_total=2500,
    epochs=5,
    adam_inner=100,
    use_lbfgs=True,
    dtype="float64",
    verbose=True,
)

# ── 3. Run forward ───────────────────────────────────────────────────
X0     = [100.0 / U_SCALE, 15.0 / U_SCALE]   # [0.5, 0.075]
T_SPAN = (0.0, 1.0)

result = adalib.run_forward(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    options=options,
)

t = result.t    # (Nt_total,)
y = result.y    # (2, Nt_total)

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
ref = sol_ref.y   # (2, Nt_total)

# ── 5. L2 relative error ─────────────────────────────────────────────
state_names = ["prey $U$", "predator $R$"]
print("\n=== L2 relative error (vs scipy RK45) ===")
for i, name in enumerate(state_names):
    err = np.linalg.norm(y[i] - ref[i]) / (np.linalg.norm(ref[i]) + 1e-12)
    print(f"  {name}: {err:.4e}")

# ── 6. Plot: all states on one axis, paper style ─────────────────────
#  colored solid = ADA,  black dashed = scipy reference
_COLORS = ["C0", "C1"]   # blue / orange

fig, ax = plt.subplots(figsize=(5, 4))

for j in range(2):
    ax.plot(t, y[j],   color=_COLORS[j], lw=1.5)
    ax.plot(t, ref[j], "k--",            lw=1.0)

ax.set_ylabel("prey, predator")
ax.set_xlabel("$t$")
ax.set_xlim(t[0], t[-1])

fig.tight_layout()
fig.savefig("lotka_forward_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → lotka_forward_result.png")
