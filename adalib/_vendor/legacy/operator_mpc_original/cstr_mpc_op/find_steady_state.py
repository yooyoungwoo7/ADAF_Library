#!/usr/bin/env python3
"""
find_steady_state.py
Q 값별 CSTR steady state T_R을 ODE 적분으로 찾는 스크립트.
cstr_mpc_op 폴더 안에서 실행:
  python find_steady_state.py
"""
import os
os.environ["PROBLEM"] = "cstr_mpc"
os.environ["BASIS"]   = "lpa"

import numpy as np
from scipy.integrate import solve_ivp
from problems.cstr_mpc_problem import (
    K0_AB, K0_BC, K0_AD, E_A_AB, E_A_BC, E_A_AD,
    H_R_AB, H_R_BC, H_R_AD, RHO, CP, CP_K,
    K_W, A_R, V_R, M_K, T_IN, C_A0_FEED,
    ALPHA_FIXED, BETA_FIXED, F_FIXED,
)

def rhs(t, x, Q):
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

x0 = [0.8, 0.5, 134.14, 130.0]  # nominal IC
T_end = 2.0  # 2시간 적분 → steady state

print(f"{'Q [kJ/h]':>12}  {'T_R_ss [°C]':>12}  {'T_K_ss [°C]':>12}  {'C_B_ss':>8}")
print("-" * 52)
for Q in [0, -500, -1000, -2000, -3000, -4000, -5000, -6000, -7000, -8000, -8500]:
    sol = solve_ivp(lambda t,y: rhs(t,y,Q), [0, T_end], x0,
                    method="BDF", rtol=1e-8, atol=1e-10, dense_output=False)
    xf = sol.y[:, -1]
    print(f"{Q:>12.0f}  {xf[2]:>12.3f}  {xf[3]:>12.3f}  {xf[1]:>8.4f}")
