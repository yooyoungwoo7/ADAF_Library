import pinn_lib 
from pinn_lib import ADAF
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from scipy.integrate import solve_ivp # 수치적 해와의 비교를 위함

#test1 : du/dt + u = 0, u(0) = 1 (1개의 ODE)
def ex_1():
    ic = [1]  # 초기 조건
    lb = 0.0
    ub = 3  

    def test1(var_list,i):
        u,u_t = var_list[0]

        if i == 0: 
            return tf.square(u_t + u)

    return ADAF.ode(test1,lb=lb,ub=ub,ic=ic,ep=10)

# test 2 : simple coupled ODE (2개의 ODE)
def ex_2():
    ic = [1.0,0.0]  # 초기 조건
    lb = 0.0
    ub = 3.0  

    def test2(var_list, i):

        # unpack
        u, u_t = var_list[0]   # u-equation variables
        v, v_t = var_list[1]   # v-equation variables

        if i == 0:
            # du/dt + u + v = 0 , u(0)=ic[0]
            res_pde = u_t + u + v
            return res_pde

        elif i == 1:
            # dv/dt + v = 0 , v(0)=ic[1]
            res_pde = v_t + v
            return res_pde
        
    return ADAF.ode(test2,ic,ep=30)

# test 3 : Lotka-Volterra Model
def ex_3():
    U = 200.0
    R = 20.0

    lb = 0.0
    ub = 1.0   # 보통 paper 예제는 [0,1], 필요하면 늘려도 됨

    ic = [100.0 / U,   # r(0)
          15.0  / U]   # p(0)

    def test3(var_list, i):
        # unpack variables
        r, r_t = var_list[0]   # prey
        p, p_t = var_list[1]   # predator

        if i == 0:
            # dr/dt = (R/U) * (2U r - 0.04 U^2 r p)
            rhs = (R / U) * (2.0 * U * r - 0.04 * U**2 * r * p)
            return r_t - rhs

        elif i == 1:
            # dp/dt = (R/U) * (0.02 U^2 r p - 1.06 U p)
            rhs = (R / U) * (0.02 * U**2 * r * p - 1.06 * U * p)
            return p_t - rhs

    return ADAF.ode(test3, lb=lb, ub=ub, ic=ic,ep=100,gamma=0.8,N_p=3000,N_m=100,)  # 너 API에 lb/ub 받는지에 따라 조정

# test 4: T/S Model
def ex_4():
    ic = [0.0, 0.0, 0.0]
    lb = 0.0
    ub = 0.05

    Re  = 6.8
    Le  = 5.0e-4
    Bl  = 3.1
    Mms = 4.8e-3
    Rms = 0.711157
    Kms = 3556.658

    # 패키지 내부와 동일한 collocation points (100개)
    L = 1.0
    gamma = 0.8
    t = np.linspace(lb, ub, 2000)
    t_train = (t - np.min(t))
    t_train = (t_train/np.max(t_train))*L*gamma
    t_train = t_train.astype('float32')
    t_train = tf.constant(t_train)

    # 입력 전압 (100 Hz, 18 Vpk)
    def vin():
        Vpk = 18.0
        two_pi = tf.constant(2.0*np.pi, tf.float32)
        return tf.constant(Vpk, tf.float32) * tf.sin(two_pi * 50.0 *t)

    def test4(var_list, i):
        cur_i, cur_i_t = var_list[0]  # i(t)
        x, x_t = var_list[1]  # x(t)
        v, v_t = var_list[2]  # v(t)=xdot(t)

        if i == 0:
            # Le * di/dt - (vin - Re*i - Bl*v) = 0
            res_pde = Le*cur_i_t - (vin() - Re*cur_i - Bl*v)
            return res_pde

        elif i == 2:
            # dx/dt - v = 0
            res_pde = x_t - v
            return res_pde

        elif i == 1:
            # Mms * dv/dt - (Bl*i - Rms*v - Kms*x) = 0
            res_pde = Mms*v_t - (Bl*cur_i - Rms*v - Kms*x)
            return res_pde

    d = ADAF.ode(test4, ic, lb=lb,ub=ub, gamma=gamma,N_p=10, N_m=100, ep=100)
    return d

#test6 : Super Capacitor (1개의 ODE)
def ex_6():
    ic = [0]  # 초기 조건
    lb = 0.0
    ub = 0.05

    # 파라미터 
    C = 1e-6
    R  = 1e3
    Vs = 5.0

    def test6(var_list,i):
        u,u_t = var_list[0]

        if i == 0: 
            return u_t - (Vs-u)/(R*C)

    f = ADAF.ode(test6,ic,lb=lb,ub=ub,ep=10,N_m = 100, N_p = 100 ,gamma=0.8)
    return f

#test7 : Euler Eq for three rigid bodies (3개의 ODE)
def ex_7():
    ic = [1, 1, 1]  # 초기 조건
    lb = 0.0
    ub = 2.5

    # 파라미터 
    I_1 = 0.2
    I_2 = 0.3
    I_3 = 0.4


    def test7(var_list,i):
        w1,w1_t = var_list[0]
        w2,w2_t = var_list[1]
        w3,w3_t = var_list[2]

        if i == 0: 
            return w1_t - ((I_2 - I_3)/(I_2*I_3))*w2*w3
        
        if i == 1: 
            return w2_t - ((I_3 - I_1)/(I_1*I_3))*w1*w3

        if i == 2: 
            return w3_t - ((I_1 - I_2)/(I_1*I_2))*w1*w2

    f = ADAF.ode(test7,ic,lb=lb,ub=ub,ep=10,N_m = 100, N_p = 100 ,gamma=0.8)
    return f

#test8 : CSTR (Continuous Stirred Tank Reactor)
def ex_8():

    k0_ab = 1.287e12
    k0_bc = 1.287e12
    k0_ad = 9.043e9

    E_A_ab = 9758.3
    E_A_bc = 9758.3
    E_A_ad = 8560.0

    H_R_ab = 4.2
    H_R_bc = -11.0
    H_R_ad = -41.85

    rho = 0.9342
    Cp  = 3.01
    Cp_k = 2.0

    A_R = 0.215
    V_R = 10.01
    m_k = 5.0
    K_w = 4032.0

    T_in = 130.0
    C_A0 = (5.7 + 4.5) / 2.0  # 5.1

    alpha = 1.0
    beta  = 1.0

 
    F     = 5.0        # [1/h]
    Q_dot = -8500.0    # [kJ/h]

    lb = 0.0
    ub = 0.1   # [h]

    ic = [0.8, 0.5,134.14,130.0]

    # -----------------------------
    # Residual function
    # -----------------------------
    def test8(var_list, i):

        # unpack states and time derivatives
        C_A, C_A_t = var_list[0]
        C_B, C_B_t = var_list[1]
        T_R, T_R_t = var_list[2]
        T_K, T_K_t = var_list[3]

        # Arrhenius terms (T in Celsius → Kelvin)
        T_Kel = T_R + 273.15

        k1 = beta * k0_ab * tf.exp(-E_A_ab / T_Kel)
        k2 =        k0_bc * tf.exp(-E_A_bc / T_Kel)
        k3 =        k0_ad * tf.exp(-alpha * E_A_ad / T_Kel)

        # -------------------------
        # C_A equation
        # -------------------------
        if i == 0:
            rhs = F * (C_A0 - C_A) - k1 * C_A - k3 * C_A**2
            return C_A_t - rhs

        # -------------------------
        # C_B equation
        # -------------------------
        elif i == 1:
            rhs = -F * C_B + k1 * C_A - k2 * C_B
            return C_B_t - rhs

        # -------------------------
        # T_R equation
        # -------------------------
        elif i == 2:
            reaction_heat = (
                k1 * C_A * H_R_ab
                + k2 * C_B * H_R_bc
                + k3 * C_A**2 * H_R_ad
            ) / (-rho * Cp)

            inflow_heat = F * (T_in - T_R)
            heat_transfer = K_w * A_R * (T_K - T_R) / (rho * Cp * V_R)

            rhs = reaction_heat + inflow_heat + heat_transfer
            return T_R_t - rhs

        # -------------------------
        # T_K equation
        # -------------------------
        elif i == 3:
            rhs = (Q_dot + K_w * A_R * (T_R - T_K)) / (m_k * Cp_k)
            return T_K_t - rhs

    f = ADAF.ode(test8,ic,lb=lb,ub=ub,ep=10,N_m = 100, N_p = 100 ,gamma=0.8)
    return f

# test9 : Batch Bioreactor (4-state system + 1 input)
def ex_9():

    # -----------------------------
    # Parameters (from PDF)
    # -----------------------------
    mu_m  = 0.02
    K_m   = 0.05
    K_i   = 5.0
    v_par = 0.004
    Y_p   = 1.2

    # uncertain parameters (PDF "Simulator" section uses these fixed values)
    Y_x  = 0.4
    S_in = 200.0

    # control input u_inp (feed flow rate of S_s)
    # PDF bounds: 0.0 <= u_inp <= 0.2
    u_inp = 0.1

    # -----------------------------
    # Time domain / IC (from PDF)
    # -----------------------------
    lb = 0.0
    ub = 100.0   # PDF closed-loop example runs n_steps=100 with t_step=1.0

    ic = [1.0, 0.5, 0.0, 120.0]   # [X_s, S_s, P_s, V_s]

    # -----------------------------
    # Residual function
    # -----------------------------
    def test9(var_list, i):

        # unpack states and time derivatives
        X_s, X_s_t = var_list[0]
        S_s, S_s_t = var_list[1]
        P_s, P_s_t = var_list[2]
        V_s, V_s_t = var_list[3]

        # mu(S_s) = mu_m * S_s / (K_m + S_s + (S_s^2 / K_i))
        mu_S = mu_m * S_s / (K_m + S_s + (S_s**2 / K_i))

        # -------------------------
        # X_s equation
        # -------------------------
        if i == 0:
            rhs = mu_S * X_s - (u_inp / V_s) * X_s
            return X_s_t - rhs

        # -------------------------
        # S_s equation
        # -------------------------
        elif i == 1:
            rhs = -mu_S * X_s / Y_x - v_par * X_s / Y_p + (u_inp / V_s) * (S_in - S_s)
            return S_s_t - rhs

        # -------------------------
        # P_s equation
        # -------------------------
        elif i == 2:
            rhs = v_par * X_s - (u_inp / V_s) * P_s
            return P_s_t - rhs

        # -------------------------
        # V_s equation
        # -------------------------
        elif i == 3:
            rhs = u_inp
            return V_s_t - rhs

    f = ADAF.ode(test9, ic, lb=lb, ub=ub, ep=10, N_m=100, N_p=100, gamma=0.8)
    return f

# test10 : Triple Tank System (3-state system + 2 inputs)
def ex_10():

    # -----------------------------
    # Parameters (from PDF)
    # -----------------------------
    A  = 0.00154       # cross-sectional area of tanks [m^2]
    g  = 9.81          # gravity [m/s^2]
    Sp = 5e-5          # cross-sectional area of pipes [m^2]

    r1 = 0.8
    r2 = 0.8
    r3 = 1.0

    # -----------------------------
    # Control inputs (constant case)
    # -----------------------------
    u1 = 1e-4          # inflow to tank 1 [m^3/s]
    u2 = 1e-4          # inflow to tank 2 [m^3/s]

    # -----------------------------
    # Time domain / IC (from PDF)
    # -----------------------------
    lb = 0.0
    ub = 200.0

    ic = [2.0, 2.8, 2.7]   # [x1, x2, x3] initial water levels

    # -----------------------------
    # Residual function
    # -----------------------------
    def test10(var_list, i):

        # unpack states and time derivatives
        x1, x1_t = var_list[0]
        x2, x2_t = var_list[1]
        x3, x3_t = var_list[2]

        # -------------------------
        # Flow rates
        # -------------------------
        q13 = r1 * Sp * tf.sign(x1 - x3) * tf.sqrt(2.0 * g * tf.abs(x1 - x3))
        q32 = r3 * Sp * tf.sign(x3 - x2) * tf.sqrt(2.0 * g * tf.abs(x3 - x2))
        q20 = r2 * Sp * tf.sqrt(2.0 * g * x2)

        # -------------------------
        # x1 equation
        # -------------------------
        if i == 0:
            rhs = (-q13 + u1) / A
            return x1_t - rhs

        # -------------------------
        # x2 equation
        # -------------------------
        elif i == 1:
            rhs = (q32 - q20 + u2) / A
            return x2_t - rhs

        # -------------------------
        # x3 equation
        # -------------------------
        elif i == 2:
            rhs = (q13 - q32) / A
            return x3_t - rhs

    f = ADAF.ode(
        test10,
        ic,
        lb=lb,
        ub=ub,
        ep=10,
        N_m=100,
        N_p=100,
        gamma=0.8
    )

    return f


# 솔버 호출
b = ex_10()


# 해석적 해와의 비교 플롯
def numerical_lotka_volterra(
    t,
    x0=10.0,
    y0=5.0,
    alpha=1.5,
    beta=1.0,
    delta=1.0,
    gamma=3.0
):

    def lv_rhs(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    sol = solve_ivp(
        lv_rhs,
        t_span=(t[0], t[-1]),
        y0=[x0, y0],
        t_eval=t,
        method="RK45",
        rtol=1e-9,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError("Lotka–Volterra ODE solver failed")

    x_num = sol.y[0]
    y_num = sol.y[1]

    return x_num, y_num

def numerical_ts_model(
    t,
    ic=(0.0, 0.0, 0.0),      # (i0, x0, v0)
    Re=6.8,
    Le=5.0e-4,
    Bl=3.1,
    Mms=4.8e-3,
    Rms=0.711157,
    Kms=3556.658,
    Vpk=18.0,
    f=50.0,                 # 너 코드: 2π*50*t
    rtol=1e-9,
    atol=1e-12,
    method="RK45"
):
    """
    Numerical reference solution for the coupled T/S (electromechanical) model.

    States:
        i(t) : coil current [A]
        x(t) : displacement [m] (or your internal unit)
        v(t) : velocity [m/s]

    ODEs:
        di/dt = (vin(t) - Re*i - Bl*v) / Le
        dx/dt = v
        dv/dt = (Bl*i - Rms*v - Kms*x) / Mms

    Returns
    -------
    i_num, x_num, v_num : numpy arrays (same shape as t)
    """

    t = np.asarray(t, dtype=float)
    i0, x0, v0 = ic

    two_pi = 2.0 * np.pi
    def vin(tt):
        return Vpk * np.sin(two_pi * f * tt)

    def rhs(tt, z):
        i, x, v = z
        di = (vin(tt) - Re*i - Bl*v) / Le
        dx = v
        dv = (Bl*i - Rms*v - Kms*x) / Mms
        return [di, dx, dv]

    sol = solve_ivp(
        rhs,
        t_span=(t[0], t[-1]),
        y0=[i0, x0, v0],
        t_eval=t,
        method=method,
        rtol=rtol,
        atol=atol
    )

    if not sol.success:
        raise RuntimeError("T/S numerical solver failed: " + str(sol.message))

    i_num = sol.y[0]
    x_num = sol.y[1]
    v_num = sol.y[2]
    return i_num, x_num, v_num

def exact_super_capacitor(t, Vs=5.0, R=1e3, C=1e-6):
    return Vs * (1.0 - np.exp(-t / (R * C)))

def numerical_euler_rigid_body(
    t,
    w10=1.0,
    w20=1.0,
    w30=1.0,
    I1=0.2,
    I2=0.3,
    I3=0.4
):

    def euler_rhs(t, w):
        w1, w2, w3 = w

        dw1dt = ((I2 - I3) / (I2 * I3)) * w2 * w3
        dw2dt = ((I3 - I1) / (I1 * I3)) * w1 * w3
        dw3dt = ((I1 - I2) / (I1 * I2)) * w1 * w2

        return [dw1dt, dw2dt, dw3dt]

    sol = solve_ivp(
        euler_rhs,
        t_span=(t[0], t[-1]),
        y0=[w10, w20, w30],
        t_eval=t,
        method="RK45",
        rtol=1e-9,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError("Euler rigid body ODE solver failed")

    w1_num = sol.y[0]
    w2_num = sol.y[1]
    w3_num = sol.y[2]

    return w1_num, w2_num, w3_num

def numerical_cstr(
    t,
    # ---- initial conditions ----
    C_A0=0.8,
    C_B0=0.5,
    T_R0=134.14,
    T_K0=130.0,
    # ---- inputs ----
    F=5.0,           # [1/h]
    Q_dot=-8500.0,   # [kJ/h]
):
    """
    Numerical reference solution for CSTR model.
    Returns:
        C_A_num, C_B_num, T_R_num, T_K_num
    """

    # ----------------------------
    # Parameters (PDF values)
    # ----------------------------
    k0_ab = 1.287e12
    k0_bc = 1.287e12
    k0_ad = 9.043e9

    E_A_ab = 9758.3
    E_A_bc = 9758.3
    E_A_ad = 8560.0

    H_R_ab = 4.2
    H_R_bc = -11.0
    H_R_ad = -41.85

    rho = 0.9342
    Cp  = 3.01
    Cp_k = 2.0

    A_R = 0.215
    V_R = 10.01
    m_k = 5.0
    K_w = 4032.0

    T_in = 130.0
    C_A_feed = (5.7 + 4.5) / 2.0  # 5.1

    alpha = 1.0
    beta  = 1.0

    # ----------------------------
    # RHS
    # ----------------------------
    def cstr_rhs(t, x):
        C_A, C_B, T_R, T_K = x

        # Kelvin for Arrhenius
        T_Kel = T_R + 273.15

        k1 = beta * k0_ab * np.exp(-E_A_ab / T_Kel)
        k2 =        k0_bc * np.exp(-E_A_bc / T_Kel)
        k3 =        k0_ad * np.exp(-alpha * E_A_ad / T_Kel)

        # Material balances
        dC_A = F * (C_A_feed - C_A) - k1 * C_A - k3 * C_A**2
        dC_B = -F * C_B + k1 * C_A - k2 * C_B

        # Energy balance (reactor)
        reaction_heat = (
            k1 * C_A * H_R_ab
            + k2 * C_B * H_R_bc
            + k3 * C_A**2 * H_R_ad
        ) / (-rho * Cp)

        inflow_heat = F * (T_in - T_R)
        heat_transfer = K_w * A_R * (T_K - T_R) / (rho * Cp * V_R)

        dT_R = reaction_heat + inflow_heat + heat_transfer

        # Jacket energy balance
        dT_K = (Q_dot + K_w * A_R * (T_R - T_K)) / (m_k * Cp_k)

        return [dC_A, dC_B, dT_R, dT_K]

    # ----------------------------
    # Solve
    # ----------------------------
    sol = solve_ivp(
        cstr_rhs,
        t_span=(t[0], t[-1]),
        y0=[C_A0, C_B0, T_R0, T_K0],
        t_eval=t,
        method="Radau",     # stiff-safe
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError("CSTR ODE solver failed")

    # ----------------------------
    # unpack & return (same style as your Euler example)
    # ----------------------------
    C_A_num = sol.y[0]
    C_B_num = sol.y[1]
    T_R_num = sol.y[2]
    T_K_num = sol.y[3]

    return C_A_num, C_B_num, T_R_num, T_K_num

def numerical_batch_bioreactor(
    t,
    # ---- initial conditions (PDF) ----
    X_s0=1.0,
    S_s0=0.5,
    P_s0=0.0,
    V_s0=120.0,
    # ---- input (PDF bound: 0.0 ~ 0.2) ----
    u_inp=0.1,
):
    """
    Numerical reference solution for Batch Bioreactor model (PDF).
    States:
        X_s : biomass
        S_s : substrate
        P_s : product
        V_s : volume

    Returns:
        X_s_num, S_s_num, P_s_num, V_s_num  (arrays aligned with t)
    """

    # ----------------------------
    # Parameters (PDF values)
    # ----------------------------
    mu_m  = 0.02
    K_m   = 0.05
    K_i   = 5.0
    v_par = 0.004
    Y_p   = 1.2

    # Simulator constants in PDF
    Y_x  = 0.4
    S_in = 200.0

    # ----------------------------
    # RHS
    # ----------------------------
    def rhs(_, x):
        X_s, S_s, P_s, V_s = x

        # avoid division issues if a solver steps weirdly
        V_s = max(V_s, 1e-12)

        # mu(S_s) = mu_m * S_s / (K_m + S_s + (S_s^2 / K_i))
        mu_S = mu_m * S_s / (K_m + S_s + (S_s**2 / K_i))

        dX_s = mu_S * X_s - (u_inp / V_s) * X_s
        dS_s = -mu_S * X_s / Y_x - v_par * X_s / Y_p + (u_inp / V_s) * (S_in - S_s)
        dP_s = v_par * X_s - (u_inp / V_s) * P_s
        dV_s = u_inp

        return [dX_s, dS_s, dP_s, dV_s]

    # ----------------------------
    # Solve
    # ----------------------------
    sol = solve_ivp(
        rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=[X_s0, S_s0, P_s0, V_s0],
        t_eval=t,
        method="Radau",   # stiff-safe
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError("Batch Bioreactor ODE solver failed")

    X_s_num = sol.y[0]
    S_s_num = sol.y[1]
    P_s_num = sol.y[2]
    V_s_num = sol.y[3]

    return X_s_num, S_s_num, P_s_num, V_s_num

def numerical_triple_tank(
    t,
    # ---- initial conditions (PDF) ----
    x1_0=2.0,
    x2_0=2.8,
    x3_0=2.7,
    # ---- constant inputs (PDF example) ----
    u1=1e-4,
    u2=1e-4,
):
    """
    Numerical reference solution for Triple Tank system (continuous).

    States:
        x1 : water level in tank 1 [m]
        x2 : water level in tank 2 [m]
        x3 : water level in tank 3 [m]

    Returns:
        x1_num, x2_num, x3_num  (arrays aligned with t)
    """

    # ----------------------------
    # Parameters (from PDF)
    # ----------------------------
    A  = 0.00154       # cross-sectional area of tanks [m^2]
    g  = 9.81          # gravity [m/s^2]
    Sp = 5e-5          # cross-sectional area of pipes [m^2]

    r1 = 0.8
    r2 = 0.8
    r3 = 1.0

    # ----------------------------
    # RHS
    # ----------------------------
    def rhs(_, x):
        x1, x2, x3 = x

        # ensure non-negative levels where needed
        x2 = max(x2, 0.0)

        q13 = r1 * Sp * np.sign(x1 - x3) * np.sqrt(2.0 * g * abs(x1 - x3))
        q32 = r3 * Sp * np.sign(x3 - x2) * np.sqrt(2.0 * g * abs(x3 - x2))
        q20 = r2 * Sp * np.sqrt(2.0 * g * x2)

        dx1 = (-q13 + u1) / A
        dx2 = (q32 - q20 + u2) / A
        dx3 = (q13 - q32) / A

        return [dx1, dx2, dx3]

    # ----------------------------
    # Solve
    # ----------------------------
    sol = solve_ivp(
        rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=[x1_0, x2_0, x3_0],
        t_eval=t,
        method="Radau",     # robust for stiff / nonsmooth systems
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError("Triple Tank ODE solver failed")

    x1_num = sol.y[0]
    x2_num = sol.y[1]
    x3_num = sol.y[2]

    return x1_num, x2_num, x3_num

x1_num, x2_num, x3_num = numerical_triple_tank(b.t)


plt.figure(figsize=(12, 3))  # 가로:세로 = 4:1
plt.plot(b.t, b.models[0].out_g_x_2(b.t_tf), label='Model #0')
plt.plot(b.t, x1_num, 'k--', label='Numerical')
plt.title('X1')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\young\Desktop\python codes\pinn_lib\results\ADA-F\\w1_10_100.png')


plt.figure(figsize=(12, 3))
plt.plot(b.t, b.models[1].out_g_x_2(b.t_tf), label='Model #1')
plt.plot(b.t, x2_num, 'k--', label='Numerical')
plt.title('X2')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\young\Desktop\python codes\pinn_lib\results\ADA-F\\w2_10_100.png')


plt.figure(figsize=(12, 3))
plt.plot(b.t, b.models[2].out_g_x_2(b.t_tf), label='Model #2')
plt.plot(b.t, x3_num, 'k--', label='Numerical')
plt.title('X3')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\young\Desktop\python codes\pinn_lib\results\ADA-F\\w3_10_100.png')

plt.show()







