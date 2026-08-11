import numpy as np
import matplotlib
matplotlib.use("Agg")
import adalib

adalib.utils.set_adalib_plot_style(style="serif")

# ── 1. System: Three-Tank Benchmark (Johansson 2000) ────────────────
#
#  세 개의 원통형 수조가 오리피스(Torricelli's law)로 연결된 시스템
#
#  States  : [h₁, h₂, h₃]   — 각 수조 수위 [cm]
#  Inputs  : [q₁, q₂]       — 펌프 유량 (탱크 1·2로 유입) [cm³/s]
#
#  Dynamics (Torricelli's law):
#    dh₁/dt = (q₁  − q₁₃) / A
#    dh₂/dt = (q₂  − q₃₂) / A
#    dh₃/dt = (q₁₃ + q₃₂ − q₃₀) / A
#
#  where q₁₃, q₃₂, q₃₀ are gravity-driven inter-tank flows.

system = adalib.get_system("triple_tank")

# ── 2. Options — Case 1: 데이터 생성 + 훈련 ──────────────────────────
# 첫 실행 시 데이터 생성 + 학습이 필요합니다 (약 3–5분).
# 이후 재실행은 아래 options_infer 사용 (빠름).
options = adalib.OperatorOptions(
    basis="lpa",

    # Data generation
    n_train=2000,
    n_val=200,
    seed=42,
    generate_data=True,
    reuse_existing_data=False,

    # Training
    train=True,
    reuse_existing_checkpoint=False,
    epochs=1000,
    batch_size=8,
    lr=3e-3,
    hidden=64,
    n_layers=2,

    # Inference
    infer=True,

    work_dir="./runs/operator_triple_tank",
    verbose=True,
)

# ── 3. Case 1 — 훈련 포함 ────────────────────────────────────────────
result = adalib.run_operator(
    system=system,
    x0=[40.0, 20.0, 30.0],       # h₁, h₂, h₃ [cm]
    t_span=(0.0, 0.5),            # informational (segment grid comes from config)
    params=[100.0, 150.0],        # q₁, q₂ [cm³/s]
    options=options,
)

print("\n=== OperatorResult (Case 1) ===")
print(f"t shape  : {result.t.shape}")
print(f"y shape  : {result.y.shape}")
print(f"t range  : [{result.t[0]:.2f}, {result.t[-1]:.2f}]")

# ── 4. Cases 2–3 — 체크포인트 재사용 ────────────────────────────────
options_infer = adalib.OperatorOptions(
    basis="lpa",
    generate_data=False,
    train=False,
    reuse_existing_checkpoint=True,
    infer=True,
    work_dir="./runs/operator_triple_tank",
    hidden=64,
    n_layers=2,
    verbose=False,
)

TEST_CASES = [
    {"x0": [40.0, 20.0, 30.0], "params": [100.0, 150.0]},
    {"x0": [25.0, 45.0, 35.0], "params": [ 80.0, 200.0]},
    {"x0": [50.0, 15.0, 10.0], "params": [ 60.0, 100.0]},
]

all_results = [result]
for tc in TEST_CASES[1:]:
    r = adalib.run_operator(
        system=system,
        x0=tc["x0"],
        t_span=(0.0, 0.5),
        params=tc["params"],
        options=options_infer,
    )
    all_results.append(r)

# ── 5. Plot rollout vs scipy reference ──────────────────────────────
state_names  = ["h1",        "h2",        "h3"]
state_labels = ["$h_1$ [cm]","$h_2$ [cm]","$h_3$ [cm]"]

# Case 1: result.plot() with solve_ivp reference (params=[q1, q2] as controls)
fig, axes, metrics = result.plot(
    reference    = "solve_ivp",
    controls     = [100.0, 150.0],   # q1, q2 [cm³/s]
    state_names  = state_labels,
    state_groups = [[0], [1], [2]],
    title        = "Three-Tank — Case 1: operator vs scipy RK45",
    save_path    = "triple_tank_operator_result.png",
    show         = False,
)
print("\nPlot saved → triple_tank_operator_result.png")
print("L2 rel errors (Case 1):", ", ".join(
    f"{n}={v:.2e}" for n, v in zip(state_names, metrics["l2_rel"][0])))

# Multi-case comparison — build column labels
col_labels = []
for tc in TEST_CASES:
    h = tc["x0"]
    q = tc["params"]
    col_labels.append(
        f"$h_1$={h[0]:.0f}, $h_2$={h[1]:.0f}, $h_3$={h[2]:.0f} cm\n"
        f"$q_1$={q[0]:.0f}, $q_2$={q[1]:.0f} cm³/s"
    )

x0_list   = [tc["x0"]     for tc in TEST_CASES]
ctrl_list = [tc["params"] for tc in TEST_CASES]   # q1, q2 treated as controls

fig2, axes2, metrics2 = adalib.utils.plot_operator_result(
    all_results,
    system      = system,
    x0          = x0_list,
    control     = ctrl_list,
    reference   = "solve_ivp",
    state_names = state_labels,
    labels      = col_labels,
    state_groups= [[0], [1], [2]],
    title       = "Three-Tank Benchmark — Operator vs Reference (3 cases)",
    save_path   = "triple_tank_operator_3cases.png",
    show        = False,
)
print("3-case plot saved → triple_tank_operator_3cases.png")
print("L2 rel errors (per case, per state):")
for i, row in enumerate(metrics2["l2_rel"]):
    print(f"  Case {i+1}: "
          + ", ".join(f"{n}={v:.2e}" for n, v in zip(state_names, row)))

# ── 6. Inference validation plot ────────────────────────────────────
fig3, axes3 = result.operator_infer(
    n_cases     = 3,
    state_names = state_labels,
    title       = "Three-Tank — LPA operator inference (3 test cases)",
    save_path   = "triple_tank_inference.png",
    show        = False,
)
print("Inference plot saved → triple_tank_inference.png")
