"""
adalib/forward/solver.py
ForwardSolver: wraps pinn_lib.ADAF_seq.solve_ivp for any ODESystem.
Uses system.rhs_tf() — pure TF operations, XLA-compatible, L-BFGS-capable.
"""
from __future__ import annotations
import sys

from ..utils.paths import legacy_forward_root as _legacy_forward_root

_FWD_PATH = str(_legacy_forward_root())
if _FWD_PATH not in sys.path:
    sys.path.insert(0, _FWD_PATH)

from pinn_lib.ADAF_seq import solve_ivp as _solve_ivp
from pinn_lib.ADAF_seq.options import BasisOptions, GridOptions, AdamOptions, LBFGSOptions


class ForwardSolver:
    """
    Forward ODE solver using ADA-F or LPA basis (pinn_lib.ADAF_seq).

    Parameters
    ----------
    system : ODESystem subclass instance (must implement rhs_tf)
    basis  : 'adaf'  (Adaptive Fourier)  or  'lpa'  (Legendre polynomial)

    What "sequential" means for the 'adaf' basis
    ---------------------------------------------
    'adaf' ("piecewise sequential", see ForwardOptions) does NOT fit the
    whole [t0, t1] interval with one network in one shot. It cuts the
    interval into `n_seg` segments (see `n_seg=` below -> GridOptions ->
    solver's internal loop) and solves them ONE AT A TIME, in temporal
    order: segment k+1 cannot start training until segment k has fully
    converged, because segment k+1's initial condition literally IS
    segment k's converged endpoint. Conceptually this is the same idea
    as classical time-marching (Euler/RK) — use the previous step's
    result as the next step's starting point — except each "step" here
    is itself a small physics-informed fit (minimize the ADA-F basis's
    residual over that segment's collocation points), not a closed-form
    update formula.

    THIS FILE only builds options and calls the solver; the actual
    sequential for-loop lives in the vendored legacy solver:
      adalib/_vendor/legacy/forward_problem_original/pinn_lib/ADAF_seq/
      core/solver_adaf.py, method `Solver.train_adam_lbfgs_piecewise`
      (see the `for k in range(self.n_seg):` loop there for the real
      per-segment continuity/warm-start/train mechanics).
    """

    def __init__(self, system, basis: str = 'adaf'):
        self.system = system
        self.basis  = basis.lower()

    def solve(
        self,
        x0,
        t_span,
        u=None,
        p=None,
        n_seg: int = 50,   # number of SEQUENTIAL segments [t0,t1] is split
                           # into — see the class docstring above
        Nt_total: int = 2000,
        gamma: float = 0.8,
        L: float = 1.0,
        N_p: int = 100,
        N_m: int = 100,
        order: int = 3,
        epochs: int = 10,
        adam_inner: int = 100,
        adam_lr: float = 1e-3,
        use_lbfgs: bool = True,
        dtype: str = 'float64',
        verbose: bool = True,
        xla_predict: bool = True,
    ):
        """
        Solve the ODE from t_span[0] to t_span[1] starting at x0.

        Returns
        -------
        solver — pinn_lib Solver object
            solver.solution.t  : ndarray (Nt_total,)
            solver.solution.y  : ndarray (n_state, Nt_total)
        """
        lb, ub = float(t_span[0]), float(t_span[1])
        system = self.system

        def ode_res(var_list, i):
            return system.rhs_tf(var_list, i, u=u, p=p)

        basis_opts = BasisOptions(
            name=self.basis, N_p=N_p, N_m=N_m,
            gamma=gamma, L=L, order=order,
        )
        grid_opts = GridOptions(
            lb=lb, ub=ub,
            Nt_total=Nt_total, n_seg=n_seg,
            gamma=gamma, L=L,
        )
        adam_opts = AdamOptions(
            epochs=epochs, inner=adam_inner,
            lr=adam_lr, dtype=dtype,
            xla_predict=xla_predict,
        )
        lbfgs_opts = LBFGSOptions(use=use_lbfgs)

        return _solve_ivp(
            ode_res=ode_res,
            ic=list(x0),
            basis=basis_opts,
            grid=grid_opts,
            adam=adam_opts,
            lbfgs=lbfgs_opts,
            verbose=verbose,
        )
