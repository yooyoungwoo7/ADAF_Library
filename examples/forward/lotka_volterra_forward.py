"""
Forward solve: Lotka-Volterra using ADAF_seq basis.
Adapted from legacy/forward_problem_original/tests_ADAF_seq_lotka.py
Run from adalib_project/ root: python examples/forward/lotka_volterra_forward.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from adalib.forward import ForwardSolver
from adalib.systems import LotkaVolterra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as scipy_ivp

# System — normalization constants matching tests_ADAF_seq_lotka.py
U_scale = 200.0
R_scale = 20.0

# Effective parameters after normalization (r = prey/U_scale, p = pred/R_scale):
#   dr/dt = (R/U)*(2*U*r - 0.04*U^2*r*p)  =>  alpha=2R=40, beta=0.04*R*U=160
#   dp/dt = (R/U)*(0.02*U^2*r*p - 1.06*U*p) => delta=0.02*R*U=80, gamma=1.06*R=21.2
system = LotkaVolterra(
    alpha = 2.0   * R_scale,           # 40.0
    beta  = 0.04  * R_scale * U_scale, # 160.0
    gamma = 1.06  * R_scale,           # 21.2
    delta = 0.02  * R_scale * U_scale, # 80.0
)
x0     = [100.0 / U_scale, 15.0 / U_scale]  # [0.5, 0.075]
t_span = (0.0, 1.0)
p      = [system.alpha, system.beta, system.gamma, system.delta]

# Forward solve
solver = ForwardSolver(system, basis='adaf')
result = solver.solve(
    x0=x0, t_span=t_span, p=p,
    n_seg=50, N_p=5, N_m=100, Nt_total=2500,
    epochs=5, adam_inner=100, use_lbfgs=True, dtype='float64',
)
sol = result.solution
t   = sol.t
y   = sol.y

# Reference
def rhs(t, x): return system.rhs(t, x, p=p)
ref = scipy_ivp(rhs, t_span, x0, t_eval=t, rtol=1e-10, atol=1e-12)

# L2 error
def l2_rel(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10)

print(f"[LotkaVolterra] L2 rel error U: {l2_rel(y[0], ref.y[0]):.4e}")
print(f"[LotkaVolterra] L2 rel error R: {l2_rel(y[1], ref.y[1]):.4e}")

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, y[0], 'C0', label='U (ADAF_seq)')
ax.plot(t, ref.y[0], 'C0--', label='U (RK45)')
ax.plot(t, y[1], 'C1', label='R (ADAF_seq)')
ax.plot(t, ref.y[1], 'C1--', label='R (RK45)')
ax.set_xlabel("t"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title("Lotka-Volterra — ADAF_seq Forward Solve")
out = os.path.join(os.path.dirname(__file__), "lotka_volterra_forward.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[DONE] {out}")
