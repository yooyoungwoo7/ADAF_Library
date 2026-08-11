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
# Euler equation for three rigid bodies
# =========================================================
ic = [1.0, 1.0, 1.0]
lb = 0.0
ub = 2.5

I_1 = 0.2
I_2 = 0.3
I_3 = 0.4


def test7(var_list, i):
    w1, w1_t = var_list[0]
    w2, w2_t = var_list[1]
    w3, w3_t = var_list[2]

    if i == 0:
        rhs = ((I_2 - I_3) / (I_2 * I_3)) * w2 * w3
        return w1_t - rhs

    elif i == 1:
        rhs = ((I_3 - I_1) / (I_1 * I_3)) * w1 * w3
        return w2_t - rhs

    elif i == 2:
        rhs = ((I_1 - I_2) / (I_1 * I_2)) * w1 * w2
        return w3_t - rhs


# =========================================================
# Options
# =========================================================

'''
basis = ADAF_seq.BasisOptions(
    name='lpa',
    order=3,
    N_p=5,
    kernel_regularizer=None,
)

'''


basis = ADAF_seq.BasisOptions(
    name='adaf',
    N_p=10,
    N_m=100,
    gamma=0.8,
    L=1.0,
)



grid = ADAF_seq.GridOptions(
    lb=lb,
    ub=ub,
    Nt_total=2500,
    n_seg = 20
)

adam = ADAF_seq.AdamOptions(
    epochs=100,
    inner=1000,
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
    ode_res=test7,
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

w1_pred = y[0]
w2_pred = y[1]
w3_pred = y[2]


# =========================================================
# Reference numerical solution
# =========================================================
def rhs(t, y):
    w1, w2, w3 = y

    dw1 = ((I_2 - I_3) / (I_2 * I_3)) * w2 * w3
    dw2 = ((I_3 - I_1) / (I_1 * I_3)) * w1 * w3
    dw3 = ((I_1 - I_2) / (I_1 * I_2)) * w1 * w2

    return [dw1, dw2, dw3]


sol_ref = solve_ivp(
    rhs,
    (lb, ub),
    ic,
    t_eval=t,
    rtol=1e-10,
    atol=1e-12
)

w1_num, w2_num, w3_num = sol_ref.y


# =========================================================
# Plot
# =========================================================
basis_label = basis.name.upper()

plt.figure(figsize=(3.5, 3))
plt.plot(t, w1_num, "--", label="w1 numerical")
plt.plot(t, w1_pred,  label=f"w1 {basis_label}")
plt.plot(t, w2_num, "--", label="w2 numerical")
plt.plot(t, w2_pred, label=f"w2 {basis_label}")
plt.plot(t, w3_num, "--", label="w3 numerical")
plt.plot(t, w3_pred, label=f"w3 {basis_label}")

plt.xlabel("t")
plt.ylabel("w1, w2, w3")
plt.tight_layout()
#plt.xlim(0, 2.5)
#plt.ylim(-1.7, 1.7)
plt.show()


# =========================================================
# L2 relative error
# =========================================================
def l2_rel(pred, ref, eps=0.0):
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return num / den


e_w1 = l2_rel(w1_pred, w1_num)
e_w2 = l2_rel(w2_pred, w2_num)
e_w3 = l2_rel(w3_pred, w3_num)

pred_all = np.stack([w1_pred, w2_pred, w3_pred], axis=1).reshape(-1)
ref_all  = np.stack([w1_num,  w2_num,  w3_num], axis=1).reshape(-1)
e_all = l2_rel(pred_all, ref_all)

print("\n[L2 relative error]")
print(f"w1:  {e_w1:.6e}")
print(f"w2:  {e_w2:.6e}")
print(f"w3:  {e_w3:.6e}")
print(f"all: {e_all:.6e}")

