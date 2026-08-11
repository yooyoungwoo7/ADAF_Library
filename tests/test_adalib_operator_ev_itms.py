import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import adalib
from adalib.mpc._generic_mpc import _NumpyLPASurrogate

adalib.utils.set_adalib_plot_style(style="serif")

# ── 1. System: EV_ITMS_Full, 1-step MPC-style Operator ─────────────────
#
#  Same lumped-RC plant as tests/test_adalib_forward_ev_itms.py, but wired
#  for Operator/MPC use instead of a single continuous forward solve:
#
#    (x0, u) -> Operator NN -> W -> one segment (duration dt) ahead -> x_end
#
#  matching the standard MPC pattern (see run_generic_mpc / _mpc_loop):
#  a control SEQUENCE is never fed to the network directly — the Operator
#  only ever learns the ONE-STEP map. A multi-step "sequence" only appears
#  later, by chaining this one-step surrogate H times during MPC/rollout.
#
#  Control u (7) = the labmate's actual actuator set (the other 6 signals
#  in their code, T_amb/v_veh/I_bat/P_mot/Q_solar/N_passenger, are
#  disturbances, not something a controller decides) — held fixed here,
#  same as mode, so the plant is autonomous given (x0, u):
#    N_comp, N_pump_bat, N_pump_drive, N_fan, N_blower, alpha_valve, P_PTC
#
#  Disturbance + mode fixed at the same operating point validated in the
#  forward test (T_amb=308.15 K, v_veh=20 m/s cruise, I_bat=40 A,
#  P_mot=40 kW, Q_solar=300 W, 2 passengers, mode M2 dual cooling).

# --- Physical parameters (Modelica EV_ITMS_Full, slides 03-07) -------------
P = dict(
    C_bat=90000.0, R_int=0.02, dVoc_dT=-0.0005, UA_bat_nom=120.0,
    N_pump_nom=3000.0, Q_nom_Ah=80.0, eta_I=0.999,
    C_motor=8000.0, C_inv=3000.0, UA_motor_nom=60.0, UA_inv_nom=40.0,
    P_rated=150000.0, eta_motor_max=0.96, eta_inv_max=0.98,
    k_motor=0.15, k_inv=0.08,
    C_cab=60000.0, UA_cab=25.0, q_human=100.0, eta_PTC=0.98,
    eps_evap_ref=1.0, N_blower_nom=100.0,
    k_comp=4.0, COP_ref=3.2, COP_kT=0.03, COP_min=1.0,
    C_loop=15000.0, UA_rad_nom=350.0, V_air_nom=30.0,
    k_veh=1.0, k_fan=0.6, k_link=200.0, ua_exp=0.8,
)

STATE_NAMES = ["T_bat", "T_mot", "T_inv", "T_cab", "T_cool_bat", "T_cool_drive", "SOC"]
CONTROL_NAMES = ["N_comp", "N_pump_bat", "N_pump_drive", "N_fan", "N_blower",
                  "alpha_valve", "P_PTC"]

# Modelica ModeMapper — [single_cab, dual_cooling, direct_connect]
MODE_FIXED = np.array([0.0, 1.0, 0.0])   # M2: dual cabin+battery cooling
D_FIXED = np.array([308.15, 20.0, 40.0, 40000.0, 300.0, 2.0])
#          T_amb,   v_veh,  I_bat, P_mot,   Q_solar, N_passenger

# Control bounds — same ranges the labmate's own random-operator sampler used
CONTROL_BOUNDS = {
    "N_comp": (0.0, 60.0), "N_pump_bat": (400.0, 3000.0),
    "N_pump_drive": (400.0, 3000.0), "N_fan": (0.0, 40.0),
    "N_blower": (0.0, 120.0), "alpha_valve": (0.0, 1.0), "P_PTC": (0.0, 5000.0),
}
# State bounds — same ranges the labmate's own random-IC sampler used
STATE_BOUNDS = {
    "T_bat": (283.15, 318.15), "T_mot": (283.15, 353.15), "T_inv": (283.15, 343.15),
    "T_cab": (263.15, 318.15), "T_cool_bat": (278.15, 318.15),
    "T_cool_drive": (278.15, 333.15), "SOC": (0.2, 0.9),
}


def _physics(state, u, d, mode, maximum, minimum, absf):
    """Modelica EV_ITMS_Full.Plant RHS, backend-agnostic (numpy here — no TF
    version needed: the Operator physics-residual loss uses system.rhs()
    directly, not rhs_tf — see adalib/mpc/_generic_mpc.py:_physics_target)."""
    T_bat, T_mot, T_inv, T_cab, T_cb, T_cd = state
    N_comp, N_pb, N_pd, N_fan, N_blw, a_valve, P_PTC = u
    T_amb, v_veh, I_bat, P_mot, Q_solar, N_pass = d
    single_cab, dual_cooling, direct_connect = mode

    r_cab = single_cab + dual_cooling * a_valve
    r_bat = dual_cooling * (1.0 - a_valve)

    def speed_scale(N):
        return (maximum(N, 1e-3) / P["N_pump_nom"]) ** P["ua_exp"]

    UA_bat_eff = P["UA_bat_nom"] * speed_scale(N_pb)
    Q_bat_gen = I_bat ** 2 * P["R_int"] - I_bat * T_bat * P["dVoc_dT"]
    Q_bat_cool = UA_bat_eff * (T_bat - T_cb)
    dT_bat = (Q_bat_gen - Q_bat_cool) / P["C_bat"]
    dSOC = -P["eta_I"] * I_bat / (P["Q_nom_Ah"] * 3600.0)

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

    P_comp = P["k_comp"] * N_comp
    COP_cool = maximum(P["COP_min"], P["COP_ref"] - P["COP_kT"] * (T_amb - 298.15))
    Q_cool_total = COP_cool * P_comp
    Q_evap_avail = r_cab * Q_cool_total
    Q_chiller = r_bat * Q_cool_total

    eps_evap = P["eps_evap_ref"] * minimum(1.0, N_blw / P["N_blower_nom"])
    Q_evap = eps_evap * Q_evap_avail
    dT_cab = (P["UA_cab"] * (T_amb - T_cab) + Q_solar
              + N_pass * P["q_human"] - Q_evap + P["eta_PTC"] * P_PTC) / P["C_cab"]

    V_air = P["k_veh"] * v_veh + P["k_fan"] * N_fan
    UA_rad_eff = P["UA_rad_nom"] * (maximum(V_air, 1e-3) / P["V_air_nom"]) ** P["ua_exp"]
    Q_link = P["k_link"] * direct_connect * (T_cd - T_cb)

    dT_cb = (Q_bat_cool - Q_chiller - UA_rad_eff * (T_cb - T_amb) + Q_link) / P["C_loop"]
    dT_cd = (P_loss_total - UA_rad_eff * (T_cd - T_amb) - Q_link) / P["C_loop"]

    return [dT_bat, dT_mot, dT_inv, dT_cab, dT_cb, dT_cd, dSOC]


def ev_rhs(t, x, u=None, p=None):
    """Autonomous given (x, u) — d and mode are fixed constants, so no
    explicit time-dependence is needed here (unlike the forward test's
    augmented-tau version, which had to fake time to drive u(t)/d(t))."""
    uu = u if u is not None else [125.0, 3000.0, 3000.0, 800.0, 100.0, 0.6, 0.0]
    return np.array(_physics(x[:6], uu, D_FIXED, MODE_FIXED, max, min, abs))


system = adalib.CallableODESystem(
    "ev_itms_full", ev_rhs,
    state_names=STATE_NAMES,
    control_names=CONTROL_NAMES,
    state_bounds=STATE_BOUNDS,
    control_bounds=CONTROL_BOUNDS,
)

# ── 2. Operator training ────────────────────────────────────────────────
#
#  dt = 30 s/segment: short enough that "hold u constant for one segment"
#  is a reasonable MPC-style assumption relative to the plant's fastest
#  time constant (~40 s, the coolant loops).

DT = 30.0
N_SEG = 20   # 20 x 30s = 10 min rollout for validation below

options = adalib.OperatorOptions(
    basis="lpa",
    n_train=2000, n_val=400, seed=42,
    generate_data=False, reuse_existing_data=True,
    train=False, reuse_existing_checkpoint=True,
    epochs=2000, batch_size=64, lr=1e-3, hidden=128, n_layers=3,
    # Bumped from defaults (N_p=8, max_order=6, Nt=20): T_bat/T_cab fit
    # visibly worse than the other 5 states at defaults — T_cab's forcing
    # depends on 3 sampled controls at once (incl. a saturating
    # min(1, N_blower/100) kink) and T_bat's time constant swings 750s-
    # 4688s depending on the sampled N_pump_bat, both harder for a
    # degree-6/8-panel Legendre basis to represent than the other states.
    lpa_n_panels=12, lpa_max_order=8, lpa_nt_seg=30,
    infer=False,
    dt=DT, n_seg=N_SEG,
    work_dir="./runs/operator_ev_itms_v2",
    verbose=True,
)

result_train = adalib.run_operator(
    system=system, x0=None, t_span=(0.0, N_SEG * DT), params=None,
    options=options,
)

surr_path = os.path.join(result_train.paths["checkpoint_dir"], "lpa_operator.npz")
surrogate = _NumpyLPASurrogate.load(surr_path)
print(f"\n[Loaded] LPA operator <- {surr_path}")

# ── 3. Validation: chain the 1-step Operator vs RK45 reference ─────────
#  (this is exactly the "sequence of control input" the user asked about —
#  it only appears here, at rollout time, as H repeated calls to the
#  1-step surrogate; the Operator itself was never trained on sequences.)

TEST_CASES = [
    {"name": "nominal cooling",
     "x0": [305.15, 313.15, 313.15, 298.15, 300.15, 305.15, 0.70],
     "u_seq": lambda k: [20.0, 3000.0, 3000.0, 20.0, 100.0, 0.6, 0.0]},
    {"name": "aggressive cooling",
     "x0": [305.15, 313.15, 313.15, 298.15, 300.15, 305.15, 0.70],
     "u_seq": lambda k: [55.0, 3000.0, 3000.0, 30.0, 120.0, 0.6, 0.0]},
    {"name": "cooling off",
     "x0": [305.15, 313.15, 313.15, 298.15, 300.15, 305.15, 0.70],
     "u_seq": lambda k: [0.0, 3000.0, 3000.0, 5.0, 0.0, 0.6, 0.0]},
    # Labmate's modelica_test_case() oscillates because its DISTURBANCES
    # (I_bat, P_mot, v_veh) swing sinusoidally at a 300 s period — we fixed
    # disturbance/mode for this experiment, so the only lever we have is
    # the CONTROL sequence. Oscillating N_comp (the main cooling actuator)
    # across segments reproduces a similarly-shaped oscillatory response
    # via the same 1-step-chained mechanism, just driven by u(t) instead
    # of d(t). Period = 240 s = 8 segments -> 2.5 cycles over the 600 s
    # rollout, comparable visual density to the labmate's plot.
    {"name": "oscillating cooling",
     "x0": [305.15, 313.15, 313.15, 298.15, 300.15, 305.15, 0.70],
     "u_seq": lambda k: [30.0 + 25.0 * np.sin(2.0 * np.pi * k * DT / 240.0),
                        3000.0, 3000.0, 20.0, 60.0, 0.6, 0.0]},
]

print("\n=== L2 relative error (LPA operator vs RK45, segment boundaries) ===")
cases = []
for tc in TEST_CASES:
    x0c = np.array(tc["x0"])

    xs_op = [x0c.copy()]
    xk = x0c.copy()
    for k in range(N_SEG):
        uk = tc["u_seq"](k)
        xk = surrogate.predict_next(xk, uk)
        xs_op.append(xk.copy())
    xs_op = np.stack(xs_op)

    xs_ref = [x0c.copy()]
    xk_ref = x0c.copy()
    for k in range(N_SEG):
        uk = tc["u_seq"](k)
        sol = solve_ivp(lambda t, x: ev_rhs(t, x, u=uk),
                        (0.0, DT), xk_ref.tolist(),
                        method="RK45", rtol=1e-8, atol=1e-10)
        xk_ref = sol.y[:, -1]
        xs_ref.append(xk_ref.copy())
    xs_ref = np.stack(xs_ref)

    l2s = (np.linalg.norm(xs_op - xs_ref, axis=0) /
           (np.linalg.norm(xs_ref, axis=0) + 1e-12))
    print(f"  {tc['name']}: " + ", ".join(f"{n}={e:.2e}" for n, e in zip(STATE_NAMES, l2s)))
    cases.append((tc["name"], xs_op, xs_ref))

# ── 4. Plot: one row per test case x 7 columns (one per state) ─────────
t_seg = np.arange(N_SEG + 1) * DT
STATE_TITLES = ["T_bat", "T_mot", "T_inv", "T_cab", "T_cool,bat", "T_cool,drive", "SOC"]
IS_TEMP = [True, True, True, True, True, True, False]
_COLORS = ["C0", "C1", "C2", "C3"]

fig, axes = plt.subplots(len(cases), 7, figsize=(24, 3 * len(cases)), sharex=True)
fig.suptitle("EV ITMS — 1-step LPA Operator (chained) vs RK45")

for r, (case_name, xs_op, xs_ref) in enumerate(cases):
    for i, (name, is_temp) in enumerate(zip(STATE_TITLES, IS_TEMP)):
        ax = axes[r, i]
        y_op  = xs_op[:, i]  - 273.15 if is_temp else xs_op[:, i]
        y_ref = xs_ref[:, i] - 273.15 if is_temp else xs_ref[:, i]
        ax.plot(t_seg, y_ref, "k--", lw=1.0,
                label="RK45" if (r == 0 and i == 0) else None)
        ax.plot(t_seg, y_op, color=_COLORS[r], lw=1.5,
                label="LPA Operator" if (r == 0 and i == 0) else None)
        if r == 0:
            ax.set_title(name)
        if i == 0:
            ax.annotate(case_name, xy=(-0.45, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", fontsize=10)
            ax.set_ylabel("degC" if is_temp else "-")
        if r == 2:
            ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)

axes[0, 0].legend(loc="best", fontsize=7)
fig.tight_layout()
fig.savefig("ev_itms_operator_result_v2.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> ev_itms_operator_result_v2.png")
