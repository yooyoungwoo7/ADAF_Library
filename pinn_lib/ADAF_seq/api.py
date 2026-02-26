from .core.solver import Solver
from .options import GridOptions, AdamOptions, LBFGSOptions
from .solution import Solution

import numpy as np
import tensorflow as tf
import time


def _to_tf_dtype(dtype):
    if isinstance(dtype, tf.DType):
        return dtype
    if isinstance(dtype, str):
        d = dtype.lower()
        if d in ["float32", "fp32", "single"]:
            return tf.float32
        if d in ["float64", "fp64", "double"]:
            return tf.float64
    raise ValueError("dtype must be 'float32'/'float64' or a tf.DType (tf.float32/tf.float64).")


def solve_ivp(
    ode_res,                 # user_residual_fn(var_list, i)
    ic,                      # initial condition list, length = ode_num

    # ---- legacy args (유지) ----
    lb=0.0,
    ub=3.0,
    ep=10,
    gamma=0.8,
    N_p=100,
    N_m=100,
    Nt_total=2000,
    n_seg=10,
    Nt_seg=None,
    L=1.0,
    adam_inner=50,
    adam_lr=1e-3,
    seed=0,
    dtype="float32",
    use_lbfgs=True,
    lbfgs_method="L-BFGS-B",
    lbfgs_options=None,
    verbose=True,

    # ---- NEW: options override ----
    grid: GridOptions = None,
    adam: AdamOptions = None,
    lbfgs: LBFGSOptions = None,
):
    # -------------------- basic checks --------------------
    if not isinstance(ic, (list, tuple)) or len(ic) == 0:
        raise ValueError("ic must be a non-empty list/tuple")
    ode_num = len(ic)

    # -------------------- options override (정리해서 1번만) --------------------
    if grid is not None:
        lb = grid.lb
        ub = grid.ub
        Nt_total = grid.Nt_total
        n_seg = grid.n_seg
        Nt_seg = grid.Nt_seg
        gamma = grid.gamma
        L = grid.L

    if adam is not None:
        ep = adam.epochs
        adam_inner = adam.inner
        adam_lr = adam.lr
        seed = adam.seed
        dtype = adam.dtype

    if lbfgs is not None:
        use_lbfgs = lbfgs.use
        lbfgs_method = lbfgs.method
        lbfgs_options = lbfgs.options

    if ub <= lb:
        raise ValueError("ub must be greater than lb")

    t_step = float(ub - lb)
    tf_dtype = _to_tf_dtype(dtype)

    # -------------------- build solver --------------------
    adaf_solver = Solver(
        user_residual_fn=ode_res,
        t_step=t_step,
        n_seg=n_seg,
        Nt_total=Nt_total,
        Nt_seg=Nt_seg,
        gamma=gamma,
        N_p=N_p,
        N_m=N_m,
        L=L,
        adam_epochs=ep,
        adam_inner=adam_inner,
        adam_lr=adam_lr,
        seed=seed,
        DTYPE=tf_dtype,
        ode_num=ode_num,
        xla_predict=(adam.xla_predict if adam is not None else True),
        xla_step=(adam.xla_step if adam is not None else False),
    )

    if verbose:
        print("\nSolver built completely.")
        print(f"- time interval: [{lb}, {ub}] (T={t_step})")
        print(f"- ode_num: {ode_num}")
        print(f"- segments: n_seg={n_seg}, Nt_total={Nt_total}, Nt_seg={adaf_solver.Nt_seg}")
        print(f"- ADAM: epochs={ep}, inner={adam_inner}, lr={adam_lr}")
        print(f"- L-BFGS: use_lbfgs={use_lbfgs}, method={lbfgs_method}")

    # -------------------- run training --------------------
    t0 = time.perf_counter()
    adaf_solver.train_adam_lbfgs_piecewise(
        ic=ic,
        use_lbfgs=use_lbfgs,
        lbfgs_method=lbfgs_method,
        lbfgs_options=lbfgs_options,
        verbose=verbose,
    )
    t1 = time.perf_counter()

    if verbose:
        print("\nODE solved (piecewise).")

    # -------------------- build Solution & attach --------------------
    # 표준 시간축 (현재 구조에서는 균일격자)
    t = np.linspace(lb, ub, adaf_solver.Nt_total).astype(np.float32)

    # solver.results_list: [ (Nt_total,), ..., ]  -> y: (ode_num, Nt_total)
    if not hasattr(adaf_solver, "results_list") or len(adaf_solver.results_list) != ode_num:
        # 혹시 solver에서 results_list를 아직 안 채우면 여기서 안전하게 만들기
        raise RuntimeError("solver.results_list was not populated. Ensure solver fills results_list after training.")

    y = np.stack([np.asarray(adaf_solver.results_list[i], dtype=np.float32) for i in range(ode_num)], axis=0)

    meta = {
        "lb": float(lb),
        "ub": float(ub),
        "t_step": float(t_step),
        "ode_num": int(ode_num),
        "Nt_total": int(adaf_solver.Nt_total),
        "n_seg": int(n_seg),
        "Nt_seg": int(adaf_solver.Nt_seg),
        "gamma": float(gamma),
        "L": float(L),
        "runtime_sec": float(t1 - t0),
        "use_lbfgs": bool(use_lbfgs),
        "lbfgs_method": str(lbfgs_method),
    }

    sol = Solution(t=t, y=y, status=0, message="success", meta=meta)

    # solver에 붙여서 반환 (유저는 solver.solution.t / solver.solution.y 로 표준 접근 가능)
    adaf_solver.solution = sol

    return adaf_solver