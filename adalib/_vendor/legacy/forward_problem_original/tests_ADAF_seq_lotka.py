import pinn_lib
from pinn_lib import ADAF_seq

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.dpi'] = 400


# =========================================================
# Lotka-Volterra equation
# =========================================================
U = 200.0
R = 20.0

ic = [100.0 / U,   # r(0): prey
      15.0  / U]   # p(0): predator

lb = 0.0
ub = 1.0


def test3(var_list, i):
    r, r_t = var_list[0]
    p, p_t = var_list[1]

    if i == 0:
        rhs = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
        return r_t - rhs

    elif i == 1:
        rhs = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)
        return p_t - rhs


# =========================================================
# Options
# =========================================================

'''
basis = ADAF_seq.BasisOptions(
    name='lpa',
    order=3,
    N_p=30,
    kernel_regularizer=None,
)

'''


basis = ADAF_seq.BasisOptions(
    name='adaf',
    N_p=5,
    N_m=100,
    gamma=0.8,
    L=1.0
)



grid = ADAF_seq.GridOptions(
    lb=lb,
    ub=ub,
    Nt_total=2500,
    n_seg=50
)

adam = ADAF_seq.AdamOptions(
    epochs=5,
    inner=100,
    lr=1e-3,
    dtype="float64"
)

lbfgs = ADAF_seq.LBFGSOptions(
    use=True
)


# =========================================================
# Solve with package
# =========================================================
solver = ADAF_seq.solve_ivp(
    ode_res=test3,
    ic=ic,
    basis=basis,
    grid=grid,
    adam=adam,
    lbfgs=lbfgs,
    verbose=True,
)

model_sol = solver.solution
t = model_sol.t
y = model_sol.y

r_pred = y[0]
p_pred = y[1]


# =========================================================
# Reference numerical solution
# =========================================================
def rhs(t, y):
    r, p = y

    dr = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
    dp = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)

    return [dr, dp]


sol_ref = solve_ivp(
    rhs,
    (lb, ub),
    ic,
    t_eval=t,
    rtol=1e-10,
    atol=1e-12
)

r_num, p_num = sol_ref.y


# =========================================================
# Plot
# =========================================================
basis_label = basis.name.upper()

plt.figure(figsize=(3.5, 3))
plt.plot(t, r_pred, label=f"r {basis_label}")
plt.plot(t, r_num, "--", label="r numerical")
plt.plot(t, p_pred, label=f"p {basis_label}")
plt.plot(t, p_num, "--", label="p numerical")


plt.xlabel("t")
plt.ylabel("prey, predator")
plt.tight_layout()
plt.xlim(0, 1.0)
plt.show()


# =========================================================
# L2 relative error
# =========================================================
def l2_rel(pred, ref, eps=0.0):
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return num / den


e_r = l2_rel(r_pred, r_num)
e_p = l2_rel(p_pred, p_num)

pred_all = np.stack([r_pred, p_pred], axis=1).reshape(-1)
ref_all  = np.stack([r_num,  p_num], axis=1).reshape(-1)
e_all = l2_rel(pred_all, ref_all)

print("\n[L2 relative error]")
print(f"r:   {e_r:.6e}")
print(f"p:   {e_p:.6e}")
print(f"all: {e_all:.6e}")