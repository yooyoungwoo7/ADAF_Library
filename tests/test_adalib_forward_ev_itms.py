import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import adalib

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
#  EV_ITMS_Full: 7-state lumped-RC EV thermal management network
#  (battery, motor, inverter, cabin, battery-coolant loop, drive-coolant
#  loop, SOC), reproducing the labmate's `modelica_test_case()` scenario
#  (Modelica EV_ITMS_Full.Test_TimeTrajectory) exactly: same x0, same
#  time-varying u(t)/d(t) profiles, same fixed mode (M2).
#
#    dT_bat/dt  = (Q_bat_gen  - Q_bat_cool)                       / C_bat
#    dT_mot/dt  = (Q_mot_loss - UA_mot,eff (T_mot - T_cool,drive)) / C_mot
#    dT_inv/dt  = (Q_inv_loss - UA_inv,eff (T_inv - T_cool,drive)) / C_inv
#    dT_cab/dt  = (UA_cab (T_amb - T_cab) + Q_solar + N_pass q_human
#                  - Q_evap + eta_PTC P_PTC)                       / C_cab
#    dT_cb/dt   = (Q_bat_cool - Q_chiller - UA_rad,eff(T_cb-T_amb)
#                  + Q_link)                                       / C_loop
#    dT_cd/dt   = (Q_loss,total - UA_rad,eff(T_cd-T_amb) - Q_link)  / C_loop
#    dSOC/dt    = -eta_I I_bat / (Q_nom_Ah * 3600)
#
#  IMPORTANT — this is a NON-AUTONOMOUS ODE (u, d vary explicitly with t),
#  but adalib's ADA-F backend never exposes real time to `rhs_tf`: the
#  solver calls `ode_res(var_list, i)` with only state values/derivatives
#  (`adalib/_vendor/legacy/forward_problem_original/pinn_lib/ADAF_seq/core
#  /solver_adaf.py`), and `adalib/forward/solver.py` forwards no time
#  either. No built-in system in this repo has ever needed non-autonomous
#  support, so there is no ready-made mechanism for it.
#
#  Workaround used here: the standard "augmented state" trick. We add an
#  8th state tau with dtau/dt = 1, tau(0) = 0, so tau tracks real time
#  *as a state ADA-F can see*. u(tau)/d(tau) are then evaluated from that
#  state instead of from an external time argument, turning the
#  non-autonomous problem into an autonomous 8-state one that fits
#  adalib's schema unmodified. `system.rhs`/`rhs_tf` below accept both a
#  7-state x (pure physics, used for the scipy reference, which has real
#  `t` natively) and an 8-state x (physics + tau, used by ADA-F).

# --- Physical parameters (Modelica EV_ITMS_Full, slides 03-07) -------------
P = dict(
    # --- Battery --------------------------------------------------------------
    C_bat=90000.0,                      # [J/K]
    R_int=0.02,                         # [ohm]
    dVoc_dT=-0.0005,                    # [V/K]
    UA_bat_nom=120.0,                   # [W/K]
    N_pump_nom=3000.0,                  # Modelica nominal input
    Q_nom_Ah=80.0,                      # [A.h]
    eta_I=0.999,
    # --- Motor / inverter -------------------------------------------------
    C_motor=8000.0, C_inv=3000.0,       # [J/K]
    UA_motor_nom=60.0, UA_inv_nom=40.0, # [W/K]
    P_rated=150000.0,                   # [W]
    eta_motor_max=0.96, eta_inv_max=0.98,
    k_motor=0.15, k_inv=0.08,
    # --- Cabin --------------------------------------------------------------
    C_cab=60000.0,                      # [J/K]
    UA_cab=25.0,                        # [W/K]
    q_human=100.0,                      # [W/person]
    eta_PTC=0.98,
    eps_evap_ref=1.0,
    N_blower_nom=100.0,
    # --- Cooling capacity conversion in Plant --------------------------------
    k_comp=4.0,                         # P_comp = k_comp*N_comp [W]
    COP_ref=3.2,
    COP_kT=0.03,                        # [1/K]
    COP_min=1.0,
    # --- Coolant loops --------------------------------------------------------
    C_loop=15000.0,                     # [J/K], both loops
    UA_rad_nom=350.0,                   # [W/K], both loops
    V_air_nom=30.0,
    k_veh=1.0,
    k_fan=0.6,
    k_link=200.0,                       # [W/K]
    ua_exp=0.8,
)

PHYS_STATE_NAMES = ["T_bat", "T_mot", "T_inv", "T_cab", "T_cool_bat", "T_cool_drive", "SOC"]
STATE_NAMES      = PHYS_STATE_NAMES + ["tau"]   # tau = augmented clock state (see above)

# Modelica ModeMapper — Python one-hot index = Modelica mode code - 1.
# columns: [single_cab, dual_cooling, direct_connect]
MODE_TABLE = np.array([
    [0, 0, 0],  # index 0 = Modelica mode 1 = M0
    [1, 0, 0],  # index 1 = Modelica mode 2 = M1: r_cab=1
    [0, 1, 0],  # index 2 = Modelica mode 3 = M2: r_cab=alpha, r_bat=1-alpha
    [0, 0, 0],  # index 3 = Modelica mode 4 = M3
    [0, 0, 1],  # index 4 = Modelica mode 5 = M4: directConnect=1
    [0, 0, 0],  # index 5 = Modelica mode 6 = M5A
    [0, 0, 0],  # index 6 = Modelica mode 7 = M5B
    [0, 0, 0],  # index 7 = Modelica mode 8 = M6
], dtype=np.float64)
MODE_FIXED = MODE_TABLE[2]   # Modelica mode 3 = M2 — matches modelica_test_case()

T_FINAL = 1800.0   # [s], matches T_FINAL in the labmate's script


# ── 2. Time-varying u(t)/d(t) — same closed-form profiles as
#      `modelica_test_case()`. That function actually samples these onto
#      73 nodes and piecewise-linearly interpolates them for solve_ivp;
#      evaluating the closed form directly (as done here) is equivalent
#      up to that interpolation's discretization error (~25 s node
#      spacing vs. the 300 s oscillation period), so it is a slightly
#      *more* accurate reference, not a different scenario.
def _u_profile(t, minimum):
    """Control u(t) = [N_comp, N_pump_bat, N_pump_drive, N_fan, N_blower,
    alpha_valve, P_PTC]."""
    N_comp = 20.0 + 30.0 * minimum(t / 200.0, 1.0)   # ramp 0-200s, then hold
    return [N_comp, 3000.0, 3000.0, 20.0, 100.0, 0.6, 0.0]


def _d_profile(t, minimum, sinf):
    """Disturbance d(t) = [T_amb, v_veh, I_bat, P_mot, Q_solar, N_passenger]."""
    two_pi = 2.0 * np.pi
    T_amb   = 303.15 + 8.0 * minimum(t / T_FINAL, 1.0)          # ramp over the horizon
    v_veh   = 25.0 + 10.0 * sinf(two_pi * t / 300.0)
    I_bat   = 40.0 + 80.0 * sinf(two_pi * t / 300.0)
    P_mot   = 30000.0 + 40000.0 * sinf(two_pi * t / 300.0)
    Q_solar = 300.0 + 300.0 * sinf(two_pi * t / T_FINAL)
    return [T_amb, v_veh, I_bat, P_mot, Q_solar, 2.0]


# ── 3. Physics core — backend-agnostic over {max,min,abs} so the exact
#      same equations (mirroring the labmate's `rhs_np` line-for-line)
#      back both the numpy path (scipy reference) and the TF-native path
#      (ADA-F training), instead of risking two independently-transcribed
#      copies drifting apart.
def _physics(state, u, d, mode, maximum, minimum, absf):
    T_bat, T_mot, T_inv, T_cab, T_cb, T_cd = state
    N_comp, N_pb, N_pd, N_fan, N_blw, a_valve, P_PTC = u
    T_amb, v_veh, I_bat, P_mot, Q_solar, N_pass = d
    single_cab, dual_cooling, direct_connect = mode

    r_cab = single_cab + dual_cooling * a_valve
    r_bat = dual_cooling * (1.0 - a_valve)

    def speed_scale(N):
        return (maximum(N, 1e-3) / P["N_pump_nom"]) ** P["ua_exp"]

    # Battery
    UA_bat_eff = P["UA_bat_nom"] * speed_scale(N_pb)
    Q_bat_gen = I_bat ** 2 * P["R_int"] - I_bat * T_bat * P["dVoc_dT"]
    Q_bat_cool = UA_bat_eff * (T_bat - T_cb)
    dT_bat = (Q_bat_gen - Q_bat_cool) / P["C_bat"]
    dSOC = -P["eta_I"] * I_bat / (P["Q_nom_Ah"] * 3600.0)

    # Motor / inverter
    load_ratio = absf(P_mot) / P["P_rated"]
    eta_motor = maximum(P["eta_motor_max"] - P["k_motor"] * load_ratio * (1.0 - load_ratio), 1e-6)
    eta_inv   = maximum(P["eta_inv_max"]   - P["k_inv"]   * load_ratio * (1.0 - load_ratio), 1e-6)
    P_loss_motor = absf(P_mot) * (1.0 / eta_motor - 1.0)
    P_loss_inv   = absf(P_mot) * (1.0 / eta_inv   - 1.0)
    P_loss_total = P_loss_motor + P_loss_inv
    UA_motor_eff = P["UA_motor_nom"] * speed_scale(N_pd)
    UA_inv_eff   = P["UA_inv_nom"]   * speed_scale(N_pd)
    dT_mot = (P_loss_motor - UA_motor_eff * (T_mot - T_cd)) / P["C_motor"]
    dT_inv = (P_loss_inv   - UA_inv_eff   * (T_inv - T_cd)) / P["C_inv"]

    # CoolingSplit + Plant conversion
    P_comp = P["k_comp"] * N_comp
    COP_cool = maximum(P["COP_min"], P["COP_ref"] - P["COP_kT"] * (T_amb - 298.15))
    Q_cool_total = COP_cool * P_comp
    Q_evap_avail = r_cab * Q_cool_total
    Q_chiller = r_bat * Q_cool_total

    # Cabin
    eps_evap = P["eps_evap_ref"] * minimum(1.0, N_blw / P["N_blower_nom"])
    Q_evap = eps_evap * Q_evap_avail
    dT_cab = (P["UA_cab"] * (T_amb - T_cab) + Q_solar
              + N_pass * P["q_human"] - Q_evap + P["eta_PTC"] * P_PTC) / P["C_cab"]

    # Coolant loops
    V_air = P["k_veh"] * v_veh + P["k_fan"] * N_fan
    UA_rad_eff = P["UA_rad_nom"] * (maximum(V_air, 1e-3) / P["V_air_nom"]) ** P["ua_exp"]
    Q_link = P["k_link"] * direct_connect * (T_cd - T_cb)

    dT_cb = (Q_bat_cool - Q_chiller - UA_rad_eff * (T_cb - T_amb) + Q_link) / P["C_loop"]
    dT_cd = (P_loss_total - UA_rad_eff * (T_cd - T_amb) - Q_link) / P["C_loop"]

    return [dT_bat, dT_mot, dT_inv, dT_cab, dT_cb, dT_cd, dSOC]


def ev_rhs(t, x, u=None, p=None):
    """Modelica EV_ITMS_Full.Plant RHS (solve_ivp / scipy callback).

    Accepts either the 7-state physical vector (scipy reference, which has
    real `t` natively) or the 8-state [physics..., tau] vector (only used
    if someone calls system.rhs directly on the augmented state).
    """
    u_t = _u_profile(t, minimum=min)
    d_t = _d_profile(t, minimum=min, sinf=np.sin)
    dxdt = _physics(x[:6], u_t, d_t, MODE_FIXED, maximum=max, minimum=min, absf=abs)
    if len(x) == 8:
        dxdt = dxdt + [1.0]   # dtau/dt = 1
    return np.array(dxdt)


def ev_rhs_tf(var_list, i, u=None, p=None):
    """TF-native residual for the augmented 8-state system (7 physics + tau).

    u(tau)/d(tau) are evaluated from the tau *state* (var_list[7][0]) since
    ADA-F never exposes real time to rhs_tf — see the module docstring.
    """
    T_bat, T_bat_t = var_list[0]
    T_mot, T_mot_t = var_list[1]
    T_inv, T_inv_t = var_list[2]
    T_cab, T_cab_t = var_list[3]
    T_cb,  T_cb_t  = var_list[4]
    T_cd,  T_cd_t  = var_list[5]
    SOC,   SOC_t   = var_list[6]
    tau,   tau_t   = var_list[7]
    dtype = T_bat.dtype

    # tf.maximum/tf.minimum/tf.abs default bare Python floats to float32
    # (e.g. the pump speeds, which never vary with tau) — that silently
    # collides with the float64 state tensors downstream, so every
    # backend op here casts both operands to the state dtype first.
    def _max(a, b): return tf.maximum(tf.cast(a, dtype), tf.cast(b, dtype))
    def _min(a, b): return tf.minimum(tf.cast(a, dtype), tf.cast(b, dtype))
    def _abs(a):    return tf.abs(tf.cast(a, dtype))
    def _sin(a):    return tf.sin(tf.cast(a, dtype))

    u_t = _u_profile(tau, minimum=_min)
    d_t = _d_profile(tau, minimum=_min, sinf=_sin)
    dxdt = _physics((T_bat, T_mot, T_inv, T_cab, T_cb, T_cd), u_t, d_t, MODE_FIXED,
                     maximum=_max, minimum=_min, absf=_abs)
    dxdt = dxdt + [tf.ones_like(tau)]   # dtau/dt = 1

    residuals = [T_bat_t, T_mot_t, T_inv_t, T_cab_t, T_cb_t, T_cd_t, SOC_t, tau_t]
    return residuals[i] - dxdt[i]


system = adalib.CallableODESystem(
    "ev_itms_full", ev_rhs, rhs_tf=ev_rhs_tf,
    state_names=STATE_NAMES,
)

# ── 4. ForwardOptions ────────────────────────────────────────────────
#  n_seg finer than the earlier constant-input version: u(t)/d(t) now
#  oscillate with a 300 s period, so each segment must be well under that
#  to be resolved (1800/60 = 30 s/segment here).
options = adalib.ForwardOptions(
    basis="adaf",
    n_seg=20,
    N_p=14,
    N_m=100,
    Nt_total=900,
    epochs=8,
    adam_inner=150,
    use_lbfgs=True,
    dtype="float64",
    verbose=True,
)

# ── 5. Run forward ───────────────────────────────────────────────────
#  x0 matches modelica_test_case() exactly, plus tau0 = 0 for the
#  augmented clock state.
X0_PHYS = [298.15, 298.15, 298.15, 295.15, 298.15, 298.15, 0.7]
X0      = X0_PHYS + [0.0]
T_SPAN  = (0.0, T_FINAL)

result = adalib.run_forward(
    system=system,
    x0=X0,
    t_span=T_SPAN,
    options=options,
)

t = result.t    # (Nt_total,)
y = result.y    # (8, Nt_total) — rows 0-6 physics, row 7 = learned tau

print("\n=== Forward result ===")
print(f"t shape  : {t.shape}")
print(f"y shape  : {y.shape}")
print(f"t range  : [{t[0]:.1f}, {t[-1]:.1f}] s")

tau_err = np.linalg.norm(y[7] - t) / (np.linalg.norm(t) + 1e-12)
print(f"\naugmented tau-state vs true t, L2 relative error: {tau_err:.4e}"
      "  (sanity check on the non-autonomous workaround)")

# ── 6. scipy RK45 reference (pure 7-state physics, real t) ────────────
sol_ref = solve_ivp(
    lambda tt, xx: ev_rhs(tt, xx),
    T_SPAN, X0_PHYS,
    method="RK45", t_eval=t, rtol=1e-10, atol=1e-12,
)
ref = sol_ref.y   # (7, Nt_total)

# ── 7. L2 relative error ─────────────────────────────────────────────
print("\n=== L2 relative error (vs scipy RK45) ===")
for i, name in enumerate(PHYS_STATE_NAMES):
    err = np.linalg.norm(y[i] - ref[i]) / (np.linalg.norm(ref[i]) + 1e-12)
    print(f"  {name}: {err:.4e}")

# ── 8. Plot: one panel per state ──────────────────────────────────────
#  black solid = scipy RK45 reference,  red dashed = ADA-F
STATE_TITLES = ["T_bat", "T_mot", "T_inv", "T_cab",
                "T_cool,bat", "T_cool,drive", "SOC"]
IS_TEMPERATURE = [True, True, True, True, True, True, False]   # SOC is unitless

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
fig.suptitle("EV ITMS — ADA-F forward vs RK45 reference")

for i, (name, is_temp) in enumerate(zip(STATE_TITLES, IS_TEMPERATURE)):
    ax = axes[i]
    y_ada = y[i] - 273.15 if is_temp else y[i]
    y_ref = ref[i] - 273.15 if is_temp else ref[i]
    ax.plot(t, y_ref, "k-",  lw=1.5, label="RK45 ref")
    ax.plot(t, y_ada, "r--", lw=1.5, label="ADA-F")
    ax.set_title(name)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Temperature [degC]" if is_temp else "SOC [-]")
    ax.grid(alpha=0.3)

axes[0].legend(loc="best", fontsize=8)
axes[-1].axis("off")   # 8th grid slot unused (only 7 physical states)

fig.tight_layout()
fig.savefig("ev_itms_forward_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> ev_itms_forward_result.png")
