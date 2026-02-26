import pinn_lib
from pinn_lib import ADAF_seq

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# test 3 : Lotka–Volterra Model
U = 200.0
R = 20.0

lb = 0.0
ub = 1.0

ic = [100.0 / U,   # r(0) prey
      15.0  / U]   # p(0) predator

def test3(var_list, i):
    r, r_t = var_list[0]
    p, p_t = var_list[1]

    if i == 0:
        rhs = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
        return r_t - rhs

    elif i == 1:
        rhs = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)
        return p_t - rhs



# 옵션들 정의
grid  = ADAF_seq.GridOptions(lb=lb, ub=ub, Nt_total=2000, n_seg=10, Nt_seg=None, gamma=0.8, L=1.0)
adam  = ADAF_seq.AdamOptions(epochs=20, inner=50, lr=1e-3, seed=0, dtype="float64", xla_predict=True, xla_step=False)
lbfgs = ADAF_seq.LBFGSOptions(use=True) 

#API 실행 
solver = ADAF_seq.solve_ivp(ode_res=test3,ic=ic,grid=grid,adam=adam,lbfgs=lbfgs,verbose=True)

# 플롯을 위한 solver 출력 정리
t = solver.solution.t                 # (Nt_total,)
y = solver.solution.y                 # (ode_num, Nt_total)  SciPy solve_ivp와 동일

r_pred = y[0]                         # prey
p_pred = y[1] 

# 3) reference numerical solution 
def rhs(t, y):
    r, p = y
    dr = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
    dp = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)
    return [dr, dp]

sol = solve_ivp(rhs, (lb, ub), [ic[0], ic[1]], t_eval=t, rtol=1e-10, atol=1e-12)
r_num = sol.y[0]
p_num = sol.y[1]

# 4) plot: time series
plt.figure(figsize=(10, 4))
plt.plot(t, r_pred,  label="r ADAF_seq")
plt.plot(t, r_num, "--", label="r numerical")
plt.plot(t, p_pred,  label="p ADAF_seq")
plt.plot(t, p_num, "--", label="p numerical")
plt.xlabel("t")
plt.ylabel("states")
plt.title("Lotka–Volterra: time series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) L2 error 출력 (각 상태별 + 전체)
def l2_rel(pred, ref, eps=1e-12):
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return num / den

e_r = l2_rel(r_pred, r_num)
e_p = l2_rel(p_pred, p_num)

pred_all = np.stack([r_pred, p_pred], axis=1).reshape(-1)
ref_all  = np.stack([r_num,  p_num],  axis=1).reshape(-1)
e_all = l2_rel(pred_all, ref_all)

print("\n[L2 relative error]")
print(f"r:   {e_r:.6e}")
print(f"p:   {e_p:.6e}")
print(f"all: {e_all:.6e}")
