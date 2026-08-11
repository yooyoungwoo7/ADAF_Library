# Operator Library — 워크플로우 (한글)

ODE 시스템을 위한 통합 physics-informed neural operator 라이브러리.
**Basis**(LPA / ADA-F)와 **Problem**(4종) 모두 단일 스위치로 전환되며 같은
학습/평가 파이프라인을 공유한다.

## 1. 디렉토리 구조

```
operator_lib/
├── config.py                    # PROBLEM_NAME, BASIS_NAME 스위치 + PROBLEM_CONFIGS 표
├── main_train.py                # 통합 학습 진입점
├── main_eval.py                 # 통합 검증/추론 진입점
├── run_all.py                   # 4 문제 일괄 실행 (sequential 또는 --parallel N)
├── requirements.txt
├── problems/
│   ├── base_problem.py          # 추상 Problem 인터페이스
│   ├── lotka_problem.py         # Lotka-Volterra (Hamiltonian 보존)
│   ├── bioreactor_problem.py    # 페드배치 Haldane (output_scale)
│   ├── cstr_problem.py          # CSTR (Arrhenius + 열 balance, output_scale)
│   ├── triple_tank_problem.py   # Torricelli 삼조 탱크
│   └── registry.py              # name → instance
├── models/
│   ├── basis.py                 # factory: get_basis_cls("lpa"|"adaf")
│   ├── lpa_basis.py             # Legendre panel basis, x(0)=x0 강제
│   ├── adaf_basis.py            # Fourier basis, x(0)=x0 AND ẋ(0)=ẋ0 강제
│   ├── operator_net.py          # vanilla MLP (zero-init + optional output_scale)
│   └── learner.py               # 범용 OperatorLearner(problem, basis_name=...)
├── data/
│   └── dataset_builder.py       # 범용 dataset builder + loader
├── utils/                       # _style, io, metrics, poly, plotting, sweep_plots
├── data_files/<problem>/        # 문제별 RK45/BDF + segment npz
└── results/<problem>/           # 문제별 학습 결과
```

## 2. 등록된 4 문제 + 2 basis

| name           | 설명                              | states / params | 솔버 | output_scale |
|----------------|-----------------------------------|-----------------|------|--------------|
| `lotka`        | Lotka-Volterra (U, R) 패밀리      | 2 / 4           | RK45 | — (RES_SCALE 균일) |
| `bioreactor`   | 페드배치 (Haldane)                | 4 / 3           | RK45 | ✓ `(0.033, 3.4, 0.025, 0.1)` |
| `cstr`         | CSTR (Arrhenius + heat balance)   | 4 / 4           | BDF  | ✓ `(0.5, 0.3, 30, 15)` |
| `triple_tank`  | Torricelli 3 탱크                 | 3 / 2           | RK45 | — (RES_SCALE 균일) |

| basis  | hard IC                          | 네트워크 출력 | 비고 |
|--------|----------------------------------|---------------|------|
| `lpa`  | `x(0) = x0`                      | `(B, state_dim, LPA_N_P)` panel 가중치 | 기본값. W → series → 해석적 적분 |
| `adaf` | `x(0) = x0` AND `ẋ(0) = f(x0,θ)` | `(B, state_dim, ADAF_N_P)` panel 가중치 | `(ξ+1)` shift로 양 IC 모두 해석적 영(0) |

`config.PROBLEM_NAME`/`BASIS_NAME` 또는 환경변수 `PROBLEM=`/`BASIS=` 또는
CLI `--problem`/`--basis` 로 전환. 한 라이브러리·한 코드 베이스로 모든 조합
처리.

## 3. 설계 철학 — single-segment training + inference-only rollout

```
[학습]   하나의 T_seg 길이 세그먼트 안에서 다양한 IC 로 학습
         loss = (ẋ_pred − f(x_pred, θ))² / RES_SCALE²   (+ cons_w · 보존량 항)
         x(0) = x_0 는 두 basis 모두 구조적으로 강제

[추론]   N_SEG 개 세그먼트를 chain — 이전 세그먼트의 x_end 를 다음 세그먼트의
         x_0 로 넣어 long horizon 복원. gradient 흐름 없음.
```

K-segment chained-gradient training (`fit_rollout`) 은 **진단용 only** —
`--rollout_epochs N` 으로 명시적으로 활성화해야 동작. 기본값 `ROLLOUT_EPOCHS=0`.

### Physics-informed 입력 (옵션)
각 문제는 `derived_features_tf(x_input)` 를 override 해서 **물리량 derived
feature** 를 입력에 자동 concat 가능. OperatorNet 은 이를 인식해서
`raw input + derived features` 를 입력으로 사용. 모든 4 문제가 6 개씩 등록:

| 문제 | 6 derived features |
|---|---|
| `lotka` | `ln(r₀/r*)`, `ln(p₀/p*)`, `V₀` (Hamiltonian), `ω·T_seg = √(αγ)·DT_SEG`, `α − β p₀`, `δ r₀ − γ` |
| `bioreactor` | `μ(Ss₀)`, `D₀ = inp/Vs₀`, `(μ−D)Xs₀`, `μSₘ + D(S_in−Ss₀)` (substrate balance), `Ss₀/K_I` (저해도), `Vs₀ + inp·DT_SEG` |
| `cstr` | `k1`, `k2`, `k3` (Arrhenius at T_R₀), `1/F` (residence time), `T_R − T_K`, reaction heat at IC |
| `triple_tank` | `Q13`, `Q32`, `Q20` at IC (signed Torricelli flows), `Q1−Q13`, `Q2+Q32−Q20`, `Q13−Q32` (탱크별 net inflow) |

문제가 override 하지 않으면 (`n_derived_features=0`) vanilla MLP 그대로 동작.

### Per-state output scaling (옵션)
`RES_SCALE` 이 비균일한 문제(bioreactor, CSTR)에서 `output_scale = RES_SCALE`
설정 시 W를 state별로 스케일링해 `∂Loss/∂W` 를 균등화. 이 없을 때 gradient
imbalance:
- bioreactor: 18,000× (Ps:Ss)
- cstr: 3,600× (T:C)

균일 RES_SCALE 문제(lotka, triple_tank)는 `output_scale = None` 유지 — Adam이
상수 스케일링을 자동 흡수하므로 효과 없음.

### Zero-init 마지막 layer
OperatorNet의 W 출력 layer 가중치/bias가 0으로 초기화 → 학습 시작 시점에서
basis가 물리적으로 안정한 baseline에서 출발:
- LPA: `x(t) = x0` (identity)
- ADA-F: `x(t) = x0 + ẋ0·t` (affine free-part만)

Stiff problem (CSTR Arrhenius `exp()`)에서 초기 residual 폭발 방지.

## 4. 학습 워크플로우

```
main_train.py [--problem <name>] [--basis lpa|adaf]
   │
   ├─ get_problem(PROBLEM_NAME)                      ← Problem 인스턴스 로드
   │
   ├─ _ensure_datasets(problem, paths, seed)
   │    ├─ build_and_save_fullcase   ← problem.sample_cases + RK45/BDF
   │    └─ build_segments_from_fullcase  ← 케이스당 1 세그먼트 랜덤 또는 N_SEG 모두
   │
   ├─ OperatorLearner(problem, hidden, n_layers, x_mean, x_std,
   │                  basis_name=BASIS_NAME)
   │    ├─ OperatorNet  (vanilla MLP, zero-init, optional output_scale)
   │    ├─ build_basis(BASIS_NAME)         : LPA 또는 ADA-F
   │    ├─ xdot0 = problem.rhs_tf(x0, θ)   : ADA-F의 ẋ(0) hard IC anchor
   │    └─ (residual + cons_w·conservation) loss
   │
   ├─ fit(...)  [single-segment physics-only training]
   │    ├─ train_step  @tf.function(jit_compile=True)
   │    ├─ cosine warmup LR (PHYSICS_LR → LR_MIN)
   │    ├─ tqdm 진행바 (phys/val/lr/초/epoch)
   │    └─ 최고 val 체크포인트: epoch_XXXX_best.weights.h5
   │
   │  (선택: --rollout_epochs N → fit_rollout(...) 진단용 K-segment 학습)
   │
   └─ save config_snapshot.json (basis, hidden, N_p 등 기록), train_summary.json,
                 history.png
```

### 실행

```bash
# 기본 (config 의 PROBLEM_NAME, BASIS_NAME)
USE_GPU=0 python main_train.py

# 문제만 전환
USE_GPU=0 python main_train.py --problem cstr

# Basis 전환
USE_GPU=0 python main_train.py --problem cstr --basis adaf
PROBLEM=triple_tank BASIS=adaf USE_GPU=0 python main_train.py

# 4 문제 일괄
USE_GPU=0 python run_all.py                       # 순차
USE_GPU=0 python run_all.py --parallel 4          # 4 병렬
USE_GPU=0 python run_all.py --basis adaf          # 모든 문제 ADA-F
```

주요 CLI 인자: `--problem`, `--basis`, `--epochs`, `--batch_size`, `--lr`,
`--hidden`, `--n_layers`, `--rollout_epochs`, `--rebuild_data`, `--seed`.

## 5. 추론·검증 워크플로우

```
main_eval.py --problem <name> --weights <ckpt> --mode {validation|custom}
   │
   ├─ _build_learner  : config_snapshot.json 에서 arch + basis 자동 로드
   │                     (학습 시 basis로 inference 강제)
   │
   ├─ validation 모드
   │    ├─ _rollout_and_report  : N_SEG 개 세그먼트 inference-only chain
   │    │    → *_traj.png, *_residual.png, *_rollout.npz
   │    ├─ plot_val_comparison  : N×state 그리드 (val 데이터)
   │    ├─ plot_param_sweeps    : problem.sweep_specs() 로 자동 생성
   │    └─ plot_random_cases    : problem.random_input() 로 4 케이스
   │
   └─ custom 모드: --x_input <input_dim 개 floats>
```

### 실행

```bash
USE_GPU=0 python main_eval.py --problem lotka \
    --weights results/lotka/<run>/checkpoints/<best>.weights.h5 \
    --mode validation --n_cases 5 --comparison_n 4 --sweep_n 8 --random_n 4
```

## 6. 새 문제 추가하기

1. `problems/<name>_problem.py` 생성 — `BaseProblem` 상속 후 다음 구현:
   - `sample_cases(n, seed)` → `(x0, theta, meta_dict)`
   - `rhs_np(t, x, theta)` (numpy)
   - `rhs_tf(x, theta)` (TF, broadcasting)
   - 옵션: `conservation_quantity(x, theta)` + `cons_w > 0`
   - 옵션: `derived_features_tf(x_input)` + `n_derived_features` +
     `derived_mean / derived_std` (physics-informed 입력 augmentation)
   - 옵션: `output_scale = RES_SCALE` — RES_SCALE 비균일 시 권장
     (`∂Loss/∂W` 균등화)
   - 옵션: `diverse_random_inputs(n, rng)` — `plot_random_cases` 가 hand-picked
     archetype 사용 (default = uniform 샘플)
   - 옵션: `extra_plot_traces()` — comparison/sweep/random plot 에 state 외
     보조 행 추가 (CSTR 의 ΔT, F, Q 같은 derived/constant trace)
   - 옵션: `segment_sampling_strategy = "all"` — 1 random seg/case 대신 모든
     segment 빌드 (CSTR이 사용 — Arrhenius transient를 항상 포함시키기 위해)
   - 옵션: `apply_train_oversampling(seg)` — train load 시 row-level 복제
     (default = identity. 현재 4문제 모두 미사용)
   - 옵션: 플롯 메타데이터 `nominal_input()`, `sweep_specs()`,
     `case_subtitle()`, `state_units()`, `state_plot_labels()`,
     `random_input()`, `time_factor`
2. `problems/registry.py` 의 `_REGISTRY` 에 한 줄 등록
3. `config.PROBLEM_CONFIGS` 에 시간/세그먼트/basis/데이터 카운트/솔버 입력

다른 코드는 변경 불필요.

## 7. 핵심 설계 포인트

- **PROBLEM × BASIS 단일 진입점** — 코드 베이스 분기 없이 4 문제 × 2 basis
  모든 조합 처리. 새 문제 추가 = `problems/` 한 파일 + `config` 한 항목 +
  `registry` 한 줄.
- **두 basis 모두 hard IC**: LPA 는 `x(0)=x0`, ADA-F 는 `x(0)=x0` + `ẋ(0)=f(x0,θ)`
  를 기저 합성 단계에서 해석적으로 강제. 네트워크는 panel 가중치만 예측,
  basis 합성식은 코드에서 분리.
- **Single-segment + inference-only rollout** 가 표준. K-segment chained
  gradient training 은 진단용으로만 사용.
- **Physics-only loss + 옵션 conservation 항**. 라벨 free.
- **JIT 컴파일** 모든 학습 step.
- **3-단 스케일링 시스템**:
  1. 입력 정규화 (`x_mean`, `x_std`, OperatorNet)
  2. 출력 스케일링 (`output_scale`, OperatorNet, 옵션)
  3. Loss residual (`RES_SCALE`, config)
  비균일 RES_SCALE 문제는 `output_scale = RES_SCALE` 로 gradient 균등화.
- **Zero-init 마지막 layer**: 학습 시작 시 W=0 → basis가 물리적으로 안정한
  baseline (`x=x0` for LPA, `x=x0+ẋ0·t` for ADA-F)에서 출발 → stiff problem
  에서 초기 NaN 방지.
