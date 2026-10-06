"""
examples/simple_api/02_forward_euler_example.py
ADALib — Feature 1: Forward ODE solving (Euler rigid body, 3-state).

Demonstrates run_forward() on a built-in 3-state ODE system.
result.forward_plot() handles the scipy reference and figure internally.

Install once:
    pip install -e .

Run from examples/simple_api/:
    python 02_forward_euler_example.py
"""
import matplotlib
matplotlib.use("Agg")
import adalib

adalib.utils.set_adalib_plot_style()

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
result = adalib.run_forward(
    system=system,
    x0=[1.0, 1.0, 1.0],
    t_span=(0.0, 2.5),
    options=options,
)

# ── 4. Plot vs scipy reference ───────────────────────────────────────
fig, axes = result.forward_plot(
    state_names = ["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"],
    title       = "Euler rigid body — ADAF vs scipy RK45  "
                  "($I_1=0.2,\\ I_2=0.3,\\ I_3=0.4$)",
    save_path   = "euler_forward_result.png",
    show        = False,
)
print("\nPlot saved → euler_forward_result.png")


