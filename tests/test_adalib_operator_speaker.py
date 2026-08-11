import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import adalib
from adalib.mpc._generic_mpc import _NumpyLPASurrogate

from utils._style import apply_style, tight_x
apply_style()

# ── 1. System: Speaker Thiele/Small (T/S) Electro-Mechanical Model ───
#
#  전기-기계 결합 3차 ODE 시스템
#
#  States:
#    I  [A]    — voice-coil 전류
#    X  [m]    — 진동판 변위
#    V  [m/s]  — 진동판 속도
#
#  Control:
#    v_in [V]  — 앰프 출력 전압  (Operator NN 입력: segment마다 상수)
#
#  T/S 파라미터 (스펙 시트 기반):
#    R_e  = 6.8 Ω,  L_e = 5e-4 H,  Bl = 3.1 T·m
#    M_ms = 4.8e-3 kg,  R_ms = 0.711157 N·s/m,  K_ms = 3556.658 N/m
#
#  Dynamics:
#    dI/dt  = (v_in − R_e·I − Bl·V) / L_e
#    dX/dt  = V
#    dV/dt  = (Bl·I − R_ms·V − K_ms·X) / M_ms
#
#  특성:
#    전기 시정수  τ_e = L_e/R_e ≈ 73.5 µs
#    기계 공진    f_s = (1/2π)√(K_ms/M_ms) ≈ 137 Hz → T ≈ 7.3 ms
#
#  ── Operator NN 입력 구조 ─────────────────────────────────────────────
#
#  매 segment k (길이 Δt)마다 NN이 보는 입력 벡터:
#
#    z_k = [I_k, X_k, V_k,  v_in_k]
#            ← 상태 (I.C) →  ← 제어 →
#
#  I.C    = [I_k, X_k, V_k]: segment 시작 시점 상태 (롤아웃 중 갱신)
#  v_in_k = Vpk · sin(2π f · k · Δt): 구간 k의 piecewise-constant 전압
#
#  R_e, L_e, Bl 등 T/S 파라미터는 훈련 데이터에 baked-in (고정 시스템)
#
#  정현파 근사 정확도:
#    Δt = 1 ms,  f = 100 Hz  →  주기당 10 segment  →  양호한 근사
#    훈련 범위: v_in ∈ [−20, 20] V  ⊃  [−18, 18] V  →  분포 이탈 없음

R_e  = 6.8;  L_e = 5e-4;  Bl = 3.1
M_ms = 4.8e-3;  R_ms = 0.711157;  K_ms = 3556.658

def speaker_rhs(_t, state, u=None, p=None):
    I, X, V = state
    v_in = float(u[0]) if u is not None else 0.0
    return [
        (v_in - R_e * I - Bl * V) / L_e,
        V,
        (Bl * I - R_ms * V - K_ms * X) / M_ms,
    ]

system = adalib.CallableODESystem(
    name="speaker_ts",
    rhs=speaker_rhs,
    state_names=["I", "X", "V"],
    control_names=["v_in"],
    state_bounds  ={"I": (-5.0, 5.0), "X": (-0.02, 0.02), "V": (-3.0, 3.0)},
    control_bounds={"v_in": (-20.0, 20.0)},
)

# ── 2. Operator 훈련 ──────────────────────────────────────────────────
#
#  Δt = 1 ms/segment,  N_SEG = 50  →  총 롤아웃 50 ms (= 5 × 100 Hz 주기)
#
#  훈련 데이터: 각 segment마다 (x_0, v_in_const) 쌍을 무작위 샘플링
#               → v_in은 구간 내 상수로 적분한 궤적을 학습
#
#  재실행 시 빠른 옵션:
#    generate_data=False, reuse_existing_data=True,
#    train=False, reuse_existing_checkpoint=True

DT    = 0.5e-3   # 1 ms/segment
N_SEG = 50     # 50 ms 롤아웃

options = adalib.OperatorOptions(
    basis="lpa",
    n_train=6000, n_val=1000, seed=42,
    generate_data=True,  reuse_existing_data=False,
    train=True,  reuse_existing_checkpoint=False,
    epochs=3000, batch_size=64, lr=1e-3, hidden=64, n_layers=3,
    # Bumped from defaults (N_p=8, max_order=6, Nt=20), same lever that
    # helped the EV system's hardest states: this T/S model is genuinely
    # stiff (electrical tau=L_e/R_e=73.5us vs a 500us segment, only ~6.8x
    # separation), so the default 8-panel/order-6 basis under-resolves the
    # fast initial transient.
    lpa_n_panels=12, lpa_max_order=8, lpa_nt_seg=30,
    infer=False,          # 훈련만 — 롤아웃은 아래 정현파로 직접 수행
    dt=DT, n_seg=N_SEG,
    work_dir="./runs/operator_speaker_ts_tight",
    verbose=True,
)

result_train = adalib.run_operator(
    system=system,
    x0=None, t_span=(0.0, N_SEG * DT), params=None,
    options=options,
)

# ── 3. 훈련된 LPA Surrogate 로드 ──────────────────────────────────────
surr_path = os.path.join(result_train.paths["checkpoint_dir"], "lpa_operator.npz")
surrogate = _NumpyLPASurrogate.load(surr_path)
print(f"\n[Loaded] LPA operator ← {surr_path}")
print("\n=== Training artifacts ===")
for a in result_train.list_artifacts():
    print(f"  {a}")

# ── 4. 정현파 롤아웃 ──────────────────────────────────────────────────
#
#  v_in(t) = Vpk · sin(2π f t)  — 스펙 시트의 최대 변위 4.5mm 이내
#
#  케이스 설계:
#    1. f = 100 Hz, x0 = [0, 0, 0]  — 정지 → 표준 오프-공진 정현파
#    2. f = 137 Hz, x0 = [0, 0, 0]  — 기계 공진 주파수 → 최대 변위
#    3. f = 100 Hz, x0 = [1, 3mm, 0.5 m/s]  — 비영 초기 조건

Vpk = 18.0   # V peak  (스펙 시트 기반)

TEST_CASES = [
    {"x0": [0.0,  0.000,  0.0],  "f": 100.0,
     "col": "$x_0=[0,0,0]$\n$f=100$ Hz (off-resonance)"},
    {"x0": [0.0,  0.000,  0.0],  "f": 137.0,
     "col": "$x_0=[0,0,0]$\n$f=137$ Hz (공진, 최대 변위)"},
    {"x0": [1.0,  0.003,  0.5],  "f": 100.0,
     "col": "$x_0=[1\\,\\mathrm{A},\\,3\\,\\mathrm{mm},\\,0.5\\,\\mathrm{m/s}]$\n$f=100$ Hz"},
]

cases = []
state_names = ["I", "X", "V"]

print("\n=== L2 상대 오차 (LPA operator vs RK45, segment boundaries) ===")

for tc in TEST_CASES:
    f   = tc["f"]
    x0c = np.array(tc["x0"], dtype=float)

    # ─ Operator 롤아웃: 구간별 piecewise-constant v_in ─────────────────
    #   k번째 segment: v_in_k = Vpk·sin(2π f · (k+0.5)·Δt)  (segment 중간값)
    #   시작값(k·Δt) 대신 중간값((k+0.5)·Δt)을 사용해 위상 오차를 1차 제거
    xs_op = [x0c.copy()]
    xk = x0c.copy()
    for k in range(N_SEG):
        v_in_k = Vpk * np.sin(2.0 * np.pi * f * (k + 0.5) * DT)
        xk = surrogate.predict_next(xk, [v_in_k])
        xs_op.append(xk.copy())
    xs_op = np.stack(xs_op)   # (N_SEG+1, 3)

    # ─ 참조해: 연속 정현파로 RK45 적분 ────────────────────────────────
    t_fine = np.linspace(0.0, N_SEG * DT, 500)
    sol = solve_ivp(
        lambda t, x, _f=f: speaker_rhs(t, x, u=[Vpk * np.sin(2.0*np.pi*_f*t)]),
        (0.0, N_SEG * DT), tc["x0"],
        method="RK45", t_eval=t_fine, rtol=1e-8, atol=1e-10,
    )
    xs_ref_fine = sol.y.T   # (500, 3)

    # ─ L2 오차: segment 경계에서 비교 ─────────────────────────────────
    t_seg = np.arange(N_SEG + 1) * DT
    xs_ref_seg = np.stack(
        [np.interp(t_seg, t_fine, xs_ref_fine[:, j]) for j in range(3)],
        axis=1,
    )
    l2s = (np.linalg.norm(xs_op - xs_ref_seg, axis=0) /
           (np.linalg.norm(xs_ref_seg, axis=0) + 1e-12))
    print(f"  f={f:.0f} Hz, x0={tc['x0']}: "
          + ", ".join(f"{n}={e:.2e}" for n, e in zip(state_names, l2s)))

    cases.append({"xs_op": xs_op, "xs_ref_fine": xs_ref_fine,
                  "t_fine": t_fine, "t_seg": t_seg})

# ── 5. Plot ──────────────────────────────────────────────────────────
#  스타일: 컬러 실선 = LPA Operator,  검은 점선 = RK45 (앞)
_COLORS = ["C0", "C1", "C2"]
state_labels = ["$I$ [A]", "$X$ [mm]", "$V$ [m/s]"]
scale        = [1.0, 1e3, 1.0]   # X: m → mm

N_CASES = len(TEST_CASES)
fig, axes = plt.subplots(
    3, N_CASES,
    figsize=(3.5 * N_CASES, 7.0),
    sharex="col",
)

for col, (tc, case) in enumerate(zip(TEST_CASES, cases)):
    xs_op       = case["xs_op"]
    xs_ref_fine = case["xs_ref_fine"]
    t_fine_ms   = case["t_fine"] * 1e3   # s → ms
    t_seg_ms    = case["t_seg"]  * 1e3

    for row in range(3):
        ax = axes[row, col]
        sc = scale[row]
        ax.plot(t_seg_ms, xs_op[:, row] * sc,
                color=_COLORS[row], lw=1.5, zorder=3)
        ax.plot(t_fine_ms, xs_ref_fine[:, row] * sc,
                "k--", lw=1.0, zorder=4)
        tight_x(ax, t_fine_ms)
        if col == 0:
            ax.set_ylabel(state_labels[row], fontsize=9)
        if row == 0:
            ax.set_title(tc["col"], fontsize=8)
        if row == 2:
            ax.set_xlabel("$t$ [ms]", fontsize=9)

# 범례
legend_elems = [
    Line2D([0], [0], color="C0", lw=1.5,         label="LPA Operator"),
    Line2D([0], [0], color="k",  lw=1.0, ls="--", label="RK45 Reference"),
]
axes[0, -1].legend(handles=legend_elems, fontsize=7, loc="upper right")

fig.suptitle(
    r"Speaker T/S Model — LPA Operator NN vs RK45"
    "\n"
    r"$v_{in}(t)=18\sin(2\pi f\,t)$ V,  $\Delta t=0.5$ ms/segment (midpoint approx.)",
    fontsize=10, y=1.01,
)
fig.tight_layout()
fig.savefig("speaker_operator_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → speaker_operator_result.png")
