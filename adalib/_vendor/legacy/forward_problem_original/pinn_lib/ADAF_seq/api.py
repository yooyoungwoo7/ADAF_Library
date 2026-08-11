from .core.solver_adaf import Solver as SolverADAF
from .core.solver_lpa import Solver as SolverLPA
from .options import BasisOptions, GridOptions, AdamOptions, LBFGSOptions
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


def solve_ivp(
    ode_res,
    ic,

    # ---- legacy args ----
    lb=0.0,
    ub=3.0,
    ep=10,
    gamma=0.8,
    N_p=100,
    N_m=100,
    Nt_total=2000,
    n_seg=50,
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

    # ---- options ----
    basis: BasisOptions = None,
    grid: GridOptions = None,
    adam: AdamOptions = None,
    lbfgs: LBFGSOptions = None,
):
    ode_num = len(ic)

    basis_name = "adaf"
    order = 3
    kernel_regularizer = None

    if basis is not None:
        basis_name = basis.name
        if hasattr(basis, "order"):
            order = basis.order
        if hasattr(basis, "kernel_regularizer"):
            kernel_regularizer = basis.kernel_regularizer
        if hasattr(basis, "N_p") and basis.N_p is not None:
            N_p = basis.N_p
        if hasattr(basis, "N_m") and basis.N_m is not None:
            N_m = basis.N_m
        if hasattr(basis, "gamma") and basis.gamma is not None:
            gamma = basis.gamma
        if hasattr(basis, "L") and basis.L is not None:
            L = basis.L

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

    t_step = float(ub - lb)
    tf_dtype = _to_tf_dtype(dtype)

    if basis_name == "adaf":
        solver = SolverADAF(
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

    if basis_name == "lpa":
        solver = SolverLPA(
            user_residual_fn=ode_res,
            t_step=t_step,
            Nt_total=Nt_total,
            N_p=N_p,
            order=order,
            lb=lb,
            ub=ub,
            adam_epochs=ep,
            adam_inner=adam_inner,
            adam_lr=adam_lr,
            seed=seed,
            DTYPE=tf_dtype,
            ode_num=ode_num,
            kernel_regularizer=kernel_regularizer,
            xla_predict=(adam.xla_predict if adam is not None else True),
            xla_step=(adam.xla_step if adam is not None else False),
        )

    if verbose:
        print("\nSolver built completely.")
        print(f"- basis: {basis_name}")
        print(f"- time interval: [{lb}, {ub}] (T={t_step})")
        print(f"- ode_num: {ode_num}")
        print(f"- Nt_total: {Nt_total}")

    t0 = time.perf_counter()

    if basis_name == "adaf":
        solver.train_adam_lbfgs_piecewise(
            ic=ic,
            use_lbfgs=use_lbfgs,
            lbfgs_method=lbfgs_method,
            lbfgs_options=lbfgs_options,
            verbose=verbose,
        )

    if basis_name == "lpa":
        solver.train(
            ic=ic,
            use_lbfgs=use_lbfgs,
            lbfgs_method=lbfgs_method,
            lbfgs_options=lbfgs_options,
            verbose=verbose,
        )

    t1 = time.perf_counter()

    if verbose:
        print("\nODE solved.")

    t = np.linspace(lb, ub, solver.Nt_total).astype(np.float32)
    y = np.stack([np.asarray(solver.results_list[i], dtype=np.float32) for i in range(ode_num)], axis=0)

    meta = {
        "lb": float(lb),
        "ub": float(ub),
        "t_step": float(t_step),
        "ode_num": int(ode_num),
        "Nt_total": int(solver.Nt_total),
        "runtime_sec": float(t1 - t0),
        "use_lbfgs": bool(use_lbfgs),
        "lbfgs_method": str(lbfgs_method),
        "basis_name": basis_name,
    }

    if basis_name == "adaf":
        meta["n_seg"] = int(n_seg)
        meta["Nt_seg"] = int(solver.Nt_seg)
        meta["gamma"] = float(gamma)
        meta["L"] = float(L)

    if basis_name == "lpa":
        meta["order"] = int(order)

    sol = Solution(
        t=t,
        y=y,
        status=0,
        message="success",
        meta=meta,
        solver=solver,
    )
    solver.solution = sol

    return solver