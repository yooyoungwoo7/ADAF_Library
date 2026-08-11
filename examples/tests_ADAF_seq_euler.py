import pinn_lib
from pinn_lib import ADAF_seq

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# test7 : Euler Eq for three rigid bodies (3개의 1st-order ODE)
ic = [1.0, 1.0, 1.0]
lb = 0.0
ub = 2.5

# parameters
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

    raise ValueError(f"Invalid equation index i={i}")


# -------------------- options 정의 --------------------
grid  = ADAF_seq.GridOptions(lb=lb, ub=ub, Nt_total=2500, n_seg=100, Nt_seg=None, gamma=0.8, L=1.0,N_p=10)
adam  = ADAF_seq.AdamOptions(epochs=10, inner=50, lr=1e-3, seed=0, dtype="float64", xla_predict=True, xla_step=False)
lbfgs = ADAF_seq.LBFGSOptions(use=True)   # 필요하면 lbfgs.options 수정

# -------------------- API 실행 --------------------
solver = ADAF_seq.solve_ivp(
    ode_res=test7,
    ic=ic,
    grid=grid,
    adam=adam,
    lbfgs=lbfgs,
    verbose=True,
)

# -------------------- solution 기반 출력 --------------------
t = solver.solution.t          # (Nt_total,)
y = solver.solution.y          # (3, Nt_total)

w1_pred = y[0]
w2_pred = y[1]
w3_pred = y[2]

# -------------------- reference numerical solution (solve_ivp) --------------------
def rhs(t, y):
    w1, w2, w3 = y
    dw1 = ((I_2 - I_3) / (I_2 * I_3)) * w2 * w3
    dw2 = ((I_3 - I_1) / (I_1 * I_3)) * w1 * w3
    dw3 = ((I_1 - I_2) / (I_1 * I_2)) * w1 * w2
    return [dw1, dw2, dw3]

sol = solve_ivp(rhs, (lb, ub), ic, t_eval=t, rtol=1e-10, atol=1e-12)
w1_num, w2_num, w3_num = sol.y

# -------------------- plot: time series --------------------
plt.plot(t, w1_num, label="w1 numerical")
plt.plot(t, w1_pred, "--", label="w1 ADAF_seq")
plt.plot(t, w2_num, label="w2 numerical")
plt.plot(t, w2_pred, "--", label="w2 ADAF_seq")
plt.plot(t, w3_num, label="w3 numerical")
plt.plot(t, w3_pred, "--", label="w3 ADAF_seq")
plt.xlabel("t")
plt.ylabel("w1, w2, w3")
plt.title("Euler rigid body (3 ODE): time series")
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- L2 error 출력 (각 상태별 + 전체) --------------------
def l2_rel(pred, ref, eps=1e-12):
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return num / den

e_w1 = l2_rel(w1_pred, w1_num)
e_w2 = l2_rel(w2_pred, w2_num)
e_w3 = l2_rel(w3_pred, w3_num)

pred_all = np.stack([w1_pred, w2_pred, w3_pred], axis=1).reshape(-1)
ref_all  = np.stack([w1_num,  w2_num,  w3_num],  axis=1).reshape(-1)
e_all = l2_rel(pred_all, ref_all)

print("\n[L2 relative error]")
print(f"w1:  {e_w1:.6e}")
print(f"w2:  {e_w2:.6e}")
print(f"w3:  {e_w3:.6e}")
print(f"all: {e_all:.6e}")