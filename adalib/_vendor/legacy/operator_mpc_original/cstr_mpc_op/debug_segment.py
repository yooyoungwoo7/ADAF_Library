#!/usr/bin/env python3
import os
os.environ["PROBLEM"] = "cstr_mpc"
os.environ["BASIS"]   = "lpa"

import numpy as np
import glob, re

from config import HIDDEN, N_LAYERS, N_SEG, T_FINAL, T0
from problems.registry import get_problem
from models.learner import OperatorLearner
from scipy.integrate import solve_ivp
from problems.cstr_mpc_problem import (
    CSTRMPCProblem, K0_AB, K0_BC, K0_AD, E_A_AB, E_A_BC, E_A_AD,
    H_R_AB, H_R_BC, H_R_AD, RHO, CP, CP_K, K_W, A_R, V_R, M_K,
    T_IN, C_A0_FEED, ALPHA_FIXED, BETA_FIXED, F_FIXED
)

def find_weight(root="results/cstr_mpc"):
    files = sorted(glob.glob(f"{root}/*/checkpoints/*best*.weights.h5"),
                   key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
                   reverse=True)
    return files[0] if files else None

def ode_rhs(t, x, Q):
    C_A, C_B, T_R, T_K = x
    Tc = T_R + 273.15
    k1 = BETA_FIXED  * K0_AB * np.exp(-E_A_AB / Tc)
    k2 =               K0_BC * np.exp(-E_A_BC / Tc)
    k3 =               K0_AD * np.exp(-ALPHA_FIXED * E_A_AD / Tc)
    dC_A = F_FIXED*(C_A0_FEED - C_A) - k1*C_A - k3*C_A*C_A
    dC_B = -F_FIXED*C_B + k1*C_A - k2*C_B
    rh   = k1*C_A*H_R_AB + k2*C_B*H_R_BC + k3*C_A*C_A*H_R_AD
    dT_R = rh/(-RHO*CP) + F_FIXED*(T_IN-T_R) + K_W*A_R*(T_K-T_R)/(RHO*CP*V_R)
    dT_K = (Q + K_W*A_R*(T_R-T_K))/(M_K*CP_K)
    return [dC_A, dC_B, dT_R, dT_K]

problem = get_problem("cstr_mpc")
learner = OperatorLearner(problem=problem, hidden=HIDDEN, n_layers=N_LAYERS, lr=1e-3)
w = find_weight()
print(f"[weight] {w}")
learner.load_weights(w)

T_seg = (T_FINAL - T0) / N_SEG

x0 = np.array([0.5, 0.4, 126.0, 120.0], dtype=np.float32)

print("\n=== 단일 세그먼트 예측 vs ODE 참조 ===")
for Q_test in [-1000, -3000, -5000, -8000]:
    # ODE 참조
    sol = solve_ivp(lambda t,y: ode_rhs(t,y,Q_test),
                    [0, T_seg], x0.tolist(),
                    method="BDF", rtol=1e-8, atol=1e-10)
    x_ref = sol.y[:, -1]

    # Operator 예측
    z = np.concatenate([x0, [Q_test]]).astype(np.float32)[None, :]
    out = learner.predict_segment(z)
    x_pred = out["x_end"][0]

    print(f"\nQ={Q_test:6.0f} kJ/h")
    print(f"  T_R  ref={x_ref[2]:.3f}  pred={x_pred[2]:.3f}  err={abs(x_pred[2]-x_ref[2]):.3f}")
    print(f"  T_K  ref={x_ref[3]:.3f}  pred={x_pred[3]:.3f}  err={abs(x_pred[3]-x_ref[3]):.3f}")
    print(f"  C_A  ref={x_ref[0]:.4f}  pred={x_pred[0]:.4f}")
