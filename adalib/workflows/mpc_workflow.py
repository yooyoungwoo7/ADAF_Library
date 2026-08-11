"""
adalib/workflows/mpc_workflow.py
run_mpc() — full closed-loop MPC workflow.

Workflow steps executed internally:
  1.  Resolve system name → MPC legacy problem name
  2.  Create nested run directory (operator/, mpc/)
  3.  Train Operator surrogate on the MPC problem variant
      (internally calls run_operator logic; no public call to run_operator)
  4.  Configure system-specific MPC objective and control bounds
  5.  Run receding-horizon MPC loop
      - Prediction model: trained Operator surrogate
      - Closed-loop plant: original ODE via scipy.integrate.solve_ivp (RK45/BDF)
  6.  Save state trajectory, control trajectory, cost history, metadata
  7.  Return MPCResult

Supported systems
-----------------
  CSTR              → tracking MPC   (minimize (T_R − T_ref)²)
  TripleTank        → tracking MPC   (minimize Σ(h_k − h_ref_k)²)
  FedBatchBioreactor → economic MPC  (maximize terminal Ps·Vs)
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import importlib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..systems.registry import get_system
from ..systems.callable_system import CallableODESystem
from ..mpc.options import MPCOptions
from ..operator.options import OperatorOptions
from ..utils.legacy_context import reload_legacy_chain
from ..utils.paths import legacy_operator_root


# ── System → legacy MPC problem name ──────────────────────────────────
_MPC_PROBLEM_MAP: dict[str, str] = {
    "cstr":                "cstr_mpc",
    "triple_tank":         "triple_tank_mpc",
    "fedbatch_bioreactor": "bioreactor",
}
_MPC_MODE_MAP: dict[str, str] = {
    "cstr":                "tracking",
    "triple_tank":         "tracking",
    "fedbatch_bioreactor": "economic",
}


# ── Result dataclass ──────────────────────────────────────────────────
@dataclass
class MPCResult:
    """Result returned by run_mpc().

    Attributes
    ----------
    t : ndarray
        Time axis of shape (n_steps+1,) [same unit as T_SEG from config].
    x : ndarray
        Closed-loop state trajectory of shape (n_steps+1, n_state).
        Plant states (true ODE plant steps).
    u : ndarray
        Applied control sequence of shape (n_steps,) or (n_steps, n_control).
    cost : ndarray
        Per-step MPC objective value of shape (n_steps,).
    operator_result : OperatorResult-like dict
        Operator training summary (train_info, paths, metadata).
    metadata : dict
        MPC configuration and timing summary.
    paths : dict
        work_dir, operator_dir, mpc_dir, checkpoint_dir, data_dir,
        trajectory_path, control_path, cost_path.
    """
    t:               np.ndarray
    x:               np.ndarray
    u:               np.ndarray
    cost:            np.ndarray
    operator_result: Dict
    metadata:        Dict
    paths:           Dict
    system:          Any = None  # stored for inference / reference support

    # ── Post-processing convenience methods ───────────────────────────

    def plot(
        self,
        save_path   = None,
        show: bool  = False,
        title       = None,
        **kwargs,
    ):
        """Plot the closed-loop MPC trajectory.

        Delegates to :func:`adalib.utils.plot_mpc_result`.

        Parameters
        ----------
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes)
        """
        from ..utils.plotting import plot_mpc_result
        return plot_mpc_result(
            self,
            save_path = save_path,
            show      = show,
            title     = title,
            **kwargs,
        )

    def operator_inference_plot(
        self,
        n_cases: int           = 4,
        cases                  = None,
        reference              = "solve_ivp",
        state_names            = None,
        control_names          = None,
        save_path              = None,
        show: bool             = False,
        title: Optional[str]   = None,
        **kwargs,
    ):
        """Multi-panel operator surrogate inference validation plot.

        Loads the LPA surrogate from ``paths["checkpoint_dir"]``,
        generates ``n_cases`` random rollouts, and plots a
        ``(n_state + n_control) × n_cases`` comparison grid.

        Parameters
        ----------
        n_cases
            Number of random test cases.
        cases
            Explicit list of case dicts with keys ``"x0"``, ``"u_seq"``.
            Overrides ``n_cases`` when provided.
        reference
            * ``"solve_ivp"`` — compare each rollout against scipy.
            * ``None`` — surrogate only.
        state_names, control_names
            Axis labels (resolved from system or metadata if omitted).
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes)
        """
        from ..utils.plotting import plot_operator_inference
        infer_cases = _run_mpc_operator_infer(
            self, n_cases=n_cases, cases=cases,
            reference=reference, **kwargs,
        )
        if state_names is None:
            state_names = (
                list(getattr(self.system, "state_names", []))
                or self.metadata.get("state_names", [])
            )
        if control_names is None:
            control_names = (
                list(getattr(self.system, "control_names", []))
                or self.metadata.get("control_names", [])
            )
        sys_name = self.metadata.get("system_name", "")
        _title = title or f"{sys_name} — MPC Operator Inference Validation"
        return plot_operator_inference(
            infer_cases,
            state_names   = state_names,
            control_names = control_names,
            save_path     = save_path,
            show          = show,
            title         = _title,
        )

    def MPC_result(
        self,
        save_path   = None,
        show: bool  = False,
        title       = None,
        **kwargs,
    ):
        """Plot the closed-loop MPC trajectory — no boilerplate needed.

        Returns
        -------
        (fig, axes)
        """
        return self.plot(save_path=save_path, show=show, title=title, **kwargs)

    def MPC_infer(
        self,
        n_cases: int = 4,
        cases        = None,
        reference    = "solve_ivp",
        state_names  = None,
        control_names = None,
        save_path    = None,
        show: bool   = False,
        title        = None,
        **kwargs,
    ):
        """Validate the operator surrogate used by MPC — no boilerplate needed.

        Loads the trained LPA surrogate from the checkpoint directory,
        generates ``n_cases`` random rollouts, and plots a validation grid.

        Parameters
        ----------
        n_cases
            Number of random test cases.
        cases
            Explicit list of case dicts (overrides n_cases).
        reference
            ``"solve_ivp"`` to compare against scipy (default), or ``None``.
        state_names, control_names
            Axis labels (auto-resolved from system/metadata if omitted).
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes)
        """
        return self.operator_inference_plot(
            n_cases       = n_cases,
            cases         = cases,
            reference     = reference,
            state_names   = state_names,
            control_names = control_names,
            save_path     = save_path,
            show          = show,
            title         = title,
            **kwargs,
        )

    def to_arrays(self):
        """Return ``(t, x, u, cost)`` as plain numpy arrays.

        Returns
        -------
        t : ndarray, shape ``(n_steps+1,)``
        x : ndarray, shape ``(n_steps+1, n_state)``
        u : ndarray, shape ``(n_steps,)`` or ``(n_steps, n_control)``
        cost : ndarray, shape ``(n_steps,)``
        """
        return (np.asarray(self.t), np.asarray(self.x),
                np.asarray(self.u), np.asarray(self.cost))

    def save_npz(self, path: Optional[str] = None) -> str:
        """Save closed-loop arrays to a compressed .npz file.

        Parameters
        ----------
        path
            Defaults to ``paths["work_dir"]/mpc_result.npz``.

        Returns
        -------
        str
            Path written to.
        """
        from ..utils.artifacts import save_npz
        if path is None:
            work = (self.paths.get("mpc_result_dir")
                    or self.paths.get("work_dir", "."))
            path = os.path.join(work, "mpc_result.npz")
        save_npz(path, t=self.t, x=self.x, u=self.u, cost=self.cost)
        return path

    def load_npz(self, path: Optional[str] = None):
        """Load a previously saved MPC .npz file.

        Parameters
        ----------
        path
            Defaults to ``paths["work_dir"]/mpc_result.npz``.

        Returns
        -------
        dict of ndarray
        """
        from ..utils.artifacts import load_npz
        if path is None:
            work = (self.paths.get("mpc_result_dir")
                    or self.paths.get("work_dir", "."))
            path = os.path.join(work, "mpc_result.npz")
        return load_npz(path)

    def list_artifacts(self):
        """List all files under ``paths['work_dir']``."""
        from ..utils.artifacts import list_run_artifacts
        work = self.paths.get("work_dir")
        return list_run_artifacts(work) if work else []


# ── Public API ─────────────────────────────────────────────────────────

def run_mpc(
    system,
    x0,
    t_span=None,
    options=None,
    **kwargs,
) -> MPCResult:
    """Full MPC workflow: operator training + closed-loop MPC.

    Parameters
    ----------
    system : str or ODESystem
        Built-in systems: ``"cstr"``, ``"triple_tank"``,
        ``"fedbatch_bioreactor"``.
    x0 : array-like
        Initial plant state.
    t_span : (t0, t1), optional
        Integration horizon hint (informational; N_SEG/DT_SEG from config
        are authoritative).
    options : MPCOptions or dict, optional
        Workflow options.  Any field can be passed as a kwarg.
    **kwargs
        Override any :class:`MPCOptions` field by name.

    Returns
    -------
    MPCResult
    """
    # ── 1. Resolve system ──────────────────────────────────────────────
    if isinstance(system, str):
        system_name = system.lower()
        sys_obj = get_system(system)
    else:
        system_name = getattr(system, "name", type(system).__name__).lower()
        sys_obj = system

    if isinstance(sys_obj, CallableODESystem):
        # Resolve options, then dispatch to the generic path.
        if options is None:
            opts = MPCOptions()
        elif isinstance(options, dict):
            opts = MPCOptions(**options)
        else:
            import copy as _copy
            opts = _copy.copy(options)
        for k, v in kwargs.items():
            if hasattr(opts, k):
                setattr(opts, k, v)
        from ..mpc._generic_mpc import run_generic_mpc
        return run_generic_mpc(sys_obj, x0, t_span, opts)

    # Resolve aliases
    from ..systems.registry import _ALIASES
    canonical = _ALIASES.get(system_name, system_name)
    system_name = canonical

    mpc_legacy_name = None
    for key, val in _MPC_PROBLEM_MAP.items():
        if key in system_name or system_name in key:
            mpc_legacy_name = val
            default_mode    = _MPC_MODE_MAP[key]
            break

    if mpc_legacy_name is None:
        raise NotImplementedError(
            f"MPC is not supported for system '{system_name}'.  "
            "Supported: " + str(sorted(_MPC_PROBLEM_MAP.keys()))
        )

    # ── 2. Resolve options ─────────────────────────────────────────────
    if options is None:
        opts = MPCOptions()
    elif isinstance(options, dict):
        opts = MPCOptions(**options)
    else:
        import copy
        opts = copy.copy(options)

    for k, v in kwargs.items():
        if hasattr(opts, k):
            setattr(opts, k, v)

    if opts.mode == "tracking" and mpc_legacy_name == "bioreactor":
        opts = _copy_opts(opts)
        opts.mode = "economic"

    # ── 3. Work directories ────────────────────────────────────────────
    work_dir     = opts.work_dir or f"./runs/{system_name}_mpc"
    work_dir     = str(Path(work_dir).resolve())
    op_work_dir  = opts.operator_work_dir or os.path.join(work_dir, "operator")
    mpc_res_dir  = opts.mpc_result_dir or os.path.join(work_dir, "mpc")

    for d in [work_dir, op_work_dir, mpc_res_dir]:
        os.makedirs(d, exist_ok=True)

    traj_dir = os.path.join(mpc_res_dir, "trajectories")
    ctrl_dir = os.path.join(mpc_res_dir, "controls")
    cost_dir = os.path.join(mpc_res_dir, "costs")
    for d in [traj_dir, ctrl_dir, cost_dir]:
        os.makedirs(d, exist_ok=True)

    paths = {
        "work_dir":          work_dir,
        # canonical spec keys
        "operator_work_dir": op_work_dir,
        "mpc_result_dir":    mpc_res_dir,
        # aliases kept for backward compatibility
        "operator_dir":      op_work_dir,
        "mpc_dir":           mpc_res_dir,
        "data_dir":          opts.data_dir or os.path.join(op_work_dir, "data"),
        "checkpoint_dir":    opts.checkpoint_dir or os.path.join(op_work_dir, "checkpoints"),
        "log_dir":           opts.log_dir or os.path.join(op_work_dir, "logs"),
        "trajectory_path":   None,
        "control_path":      None,
        "cost_path":         None,
    }

    if opts.verbose:
        print(f"[run_mpc] system={system_name}  mpc_problem={mpc_legacy_name}  "
              f"mode={opts.mode}")
        print(f"[run_mpc] work_dir={work_dir}")

    # ── 4. Ensure legacy path & reload ────────────────────────────────
    _ensure_legacy_on_path()
    reload_legacy_chain(mpc_legacy_name, opts.basis)

    import config as _cfg
    import problems.registry as _preg
    import data.dataset_builder as _db
    import models.learner as _ml

    problem = _preg.get_problem(mpc_legacy_name)

    # ── 5. Operator training ───────────────────────────────────────────
    op_result_info: Dict = {}
    learner = None
    best_ckpt_path = None

    data_dir = paths["data_dir"]
    ckpt_dir = paths["checkpoint_dir"]
    log_dir  = paths["log_dir"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    prefix          = mpc_legacy_name
    train_full_path = os.path.join(data_dir, f"{prefix}_train_fullcases.npz")
    val_full_path   = os.path.join(data_dir, f"{prefix}_val_fullcases.npz")
    train_seg_path  = os.path.join(data_dir, f"{prefix}_train_segments.npz")
    val_seg_path    = os.path.join(data_dir, f"{prefix}_val_segments.npz")

    need_data = (opts.generate_data and not opts.reuse_existing_data
                 and opts.train_operator)

    if need_data or opts.force_rebuild_data:
        if opts.verbose:
            print(f"[run_mpc] Generating {opts.n_train} train / "
                  f"{opts.n_val} val operator trajectories...")
        _db.build_and_save_fullcase(problem, train_full_path, opts.n_train, seed=opts.seed)
        _db.build_and_save_fullcase(problem, val_full_path,   opts.n_val,   seed=opts.seed + 1)

        if opts.verbose:
            print("[run_mpc] Slicing into segments...")
        train_seg = _db.build_segments_from_fullcase(
            problem, train_full_path, seed=opts.seed + 100)
        X_mean, X_std = _db.compute_input_stats(train_seg["X"])
        _db.save_segments(train_seg_path, train_seg, X_mean=X_mean, X_std=X_std)

        val_seg = _db.build_segments_from_fullcase(
            problem, val_full_path, seed=opts.seed + 101)
        _db.save_segments(val_seg_path, val_seg, X_mean=X_mean, X_std=X_std)

    elif opts.reuse_existing_data:
        for p_ in [train_seg_path, val_seg_path]:
            if not os.path.exists(p_):
                raise FileNotFoundError(
                    f"reuse_existing_data=True but missing:\n  {p_}\n"
                    "Set reuse_existing_data=False to regenerate."
                )

    train = _db.load_segments(train_seg_path)
    val   = _db.load_segments(val_seg_path)

    if opts.train_operator and not opts.reuse_existing_operator:
        if opts.verbose:
            print(f"[run_mpc] Training Operator surrogate  "
                  f"epochs={opts.epochs}  hidden={opts.hidden}  "
                  f"n_layers={opts.n_layers}")

        train_ds = _db.make_tf_dataset(
            train, batch_size=opts.batch_size, shuffle=True, problem=problem)
        val_ds   = _db.make_tf_dataset(
            val,   batch_size=opts.batch_size, shuffle=False)

        learner = _ml.OperatorLearner(
            problem  = problem,
            hidden   = opts.hidden,
            n_layers = opts.n_layers,
            lr       = opts.lr,
            x_mean   = train.get("X_mean"),
            x_std    = train.get("X_std"),
        )

        op_train_info = learner.fit(
            train_ds, val_ds,
            epochs          = opts.epochs,
            checkpoint_dir  = ckpt_dir,
            use_lr_schedule = opts.use_lr_schedule,
        )
        best_ckpt_path = op_train_info.get("best_checkpoint")
        if best_ckpt_path:
            learner.load_weights(best_ckpt_path)
            if opts.verbose:
                print(f"[run_mpc] Loaded best checkpoint: {best_ckpt_path}")

        op_result_info = {
            "train_info":      op_train_info,
            "best_checkpoint": best_ckpt_path,
            "best_val_loss":   op_train_info.get("best_val_phys_loss"),
        }

    elif opts.reuse_existing_operator:
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*best*.weights.h5")))
        if not ckpt_files:
            raise FileNotFoundError(
                f"reuse_existing_operator=True but no '*best*.weights.h5' "
                f"found in:\n  {ckpt_dir}\n"
                "Set reuse_existing_operator=False to retrain."
            )
        best_ckpt_path = ckpt_files[-1]
        learner = _ml.OperatorLearner(
            problem  = problem,
            hidden   = opts.hidden,
            n_layers = opts.n_layers,
            lr       = opts.lr,
            x_mean   = train.get("X_mean"),
            x_std    = train.get("X_std"),
        )
        learner.load_weights(best_ckpt_path)
        op_result_info = {"best_checkpoint": best_ckpt_path, "reused": True}

        if opts.verbose:
            print(f"[run_mpc] Reused operator checkpoint: {best_ckpt_path}")

    if learner is None:
        raise RuntimeError(
            "Operator learner was not created.  "
            "Ensure train_operator=True or reuse_existing_operator=True."
        )

    # ── 6. MPC loop ────────────────────────────────────────────────────
    if not opts.run_closed_loop:
        metadata = _build_mpc_meta(system_name, mpc_legacy_name, opts,
                                   best_ckpt_path, n_steps=opts.n_steps,
                                   elapsed=0.0)
        _write_meta(work_dir, metadata)
        return MPCResult(
            t=np.array([]), x=np.array([]), u=np.array([]), cost=np.array([]),
            operator_result=op_result_info,
            metadata=metadata,
            paths=paths,
            system=sys_obj,
        )

    x0_arr = np.asarray(x0, dtype=np.float32)
    t_seg  = float(_cfg.DT_SEG)    # segment length in hours
    # convert to display unit (minutes for CSTR, seconds for triple_tank)
    time_factor = float(getattr(problem, "time_factor", 1.0))
    t_seg_display = t_seg * time_factor

    if opts.verbose:
        print(f"\n[run_mpc] Closed-loop MPC  n_steps={opts.n_steps}  "
              f"T_seg={t_seg_display:.2f} {getattr(problem, 'time_unit', '')}")

    t0_wall = time.perf_counter()

    solver_stats: Dict = {}
    use_surrogate_loop = (
        mpc_legacy_name in ("cstr_mpc", "triple_tank_mpc", "bioreactor")
        and (getattr(opts, "gradient", None) is not None
             or str(getattr(opts, "optimizer", "SLSQP")).upper()
             in ("CEM", "MPPI"))
    )

    if use_surrogate_loop:
        from ..mpc._surrogate_mpc import run_builtin_surrogate_mpc
        t_arr, x_arr, u_arr, J_arr, solver_stats = run_builtin_surrogate_mpc(
            learner, problem, mpc_legacy_name, x0_arr, opts, _cfg)
    elif mpc_legacy_name == "cstr_mpc":
        t_arr, x_arr, u_arr, J_arr = _run_cstr_tracking_mpc(
            learner, problem, x0_arr, opts, _cfg)
    elif mpc_legacy_name == "triple_tank_mpc":
        t_arr, x_arr, u_arr, J_arr = _run_triple_tank_tracking_mpc(
            learner, problem, x0_arr, opts, _cfg)
    elif mpc_legacy_name == "bioreactor":
        t_arr, x_arr, u_arr, J_arr = _run_bioreactor_empc(
            learner, problem, x0_arr, opts, _cfg)
    else:
        raise NotImplementedError(f"No MPC loop implemented for {mpc_legacy_name}")

    elapsed = time.perf_counter() - t0_wall

    # ── 7. Save results ────────────────────────────────────────────────
    traj_path = os.path.join(traj_dir, "trajectory.npz")
    ctrl_path = os.path.join(ctrl_dir, "controls.npz")
    cost_path = os.path.join(cost_dir, "costs.npz")

    np.savez_compressed(traj_path, t=t_arr, x=x_arr, x0=x0_arr)
    np.savez_compressed(ctrl_path, u=u_arr)
    np.savez_compressed(cost_path, cost=J_arr)

    # Combined closed_loop.npz in mpc_result_dir for easy access
    cl_path = os.path.join(mpc_res_dir, "closed_loop.npz")
    np.savez_compressed(cl_path, t=t_arr, x=x_arr, u=u_arr, cost=J_arr, x0=x0_arr)

    paths["trajectory_path"]  = traj_path
    paths["control_path"]     = ctrl_path
    paths["cost_path"]        = cost_path
    paths["closed_loop_path"] = cl_path

    if opts.verbose:
        print(f"\n[run_mpc] Completed {opts.n_steps} steps in {elapsed:.2f}s "
              f"({elapsed/opts.n_steps*1000:.1f} ms/step)")
        print(f"[run_mpc] Results: {mpc_res_dir}")

    # ── 8. Metadata ────────────────────────────────────────────────────
    metadata = _build_mpc_meta(system_name, mpc_legacy_name, opts,
                                best_ckpt_path, opts.n_steps, elapsed)
    metadata["final_state"] = x_arr[-1].tolist() if len(x_arr) > 0 else None
    metadata.update(solver_stats)
    _write_meta(work_dir, metadata)

    return MPCResult(
        t=t_arr,
        x=x_arr,
        u=u_arr,
        cost=J_arr,
        operator_result=op_result_info,
        metadata=metadata,
        paths=paths,
        system=sys_obj,
    )


# ── MPC loop implementations ──────────────────────────────────────────

def _predict_next_scalar(learner, xk: np.ndarray, u_scalar: float,
                          x_min=None, x_max=None) -> np.ndarray:
    """Single-segment operator prediction with 1-D scalar control appended."""
    z = np.concatenate([xk, [float(u_scalar)]]).astype(np.float32)[None, :]
    out = learner.predict_segment(z)
    xn  = out["x_end"][0].astype(np.float32)
    if x_min is not None:
        xn = np.clip(xn, x_min, x_max)
    return xn


def _predict_next_vec(learner, xk: np.ndarray, u_vec: np.ndarray,
                      x_min=None, x_max=None) -> np.ndarray:
    """Single-segment operator prediction with multi-D control appended."""
    z = np.concatenate([xk, np.asarray(u_vec, dtype=np.float32)]).astype(np.float32)[None, :]
    out = learner.predict_segment(z)
    xn  = out["x_end"][0].astype(np.float32)
    if x_min is not None:
        xn = np.clip(xn, x_min, x_max)
    return xn


def _plant_step_scipy(problem, xk: np.ndarray, u_scalar: float,
                       t_seg: float, ref_solver: str = "BDF") -> np.ndarray:
    """Advance the true ODE plant one segment via scipy solve_ivp."""
    from scipy.integrate import solve_ivp
    import numpy as np

    # Build theta for ODE: [Q] for cstr_mpc (only scalar)
    theta = np.array([float(u_scalar)], dtype=np.float64)

    def ode(t, y):
        return problem._rhs(t, y, theta[0]) if hasattr(problem, "_rhs") else \
               problem.rhs_np(t, y, theta)

    sol = solve_ivp(
        ode,
        t_span=(0.0, t_seg),
        y0=np.asarray(xk, dtype=np.float64),
        method=ref_solver,
        rtol=1e-6, atol=1e-8,
    )
    return sol.y[:, -1].astype(np.float32)


def _plant_step_scipy_vec(problem, xk: np.ndarray, u_vec: np.ndarray,
                           t_seg: float, ref_solver: str = "RK45") -> np.ndarray:
    """Advance the true ODE plant with multi-D control via scipy solve_ivp."""
    from scipy.integrate import solve_ivp

    theta = np.asarray(u_vec, dtype=np.float64)

    def ode(t, y):
        return problem.rhs_np(t, y, theta)

    sol = solve_ivp(
        ode,
        t_span=(0.0, t_seg),
        y0=np.asarray(xk, dtype=np.float64),
        method=ref_solver,
        rtol=1e-6, atol=1e-8,
    )
    return sol.y[:, -1].astype(np.float32)


# ── CSTR tracking MPC ─────────────────────────────────────────────────

def _run_cstr_tracking_mpc(learner, problem, x0, opts, cfg):
    """1-step greedy tracking MPC for CSTR.

    Minimizes (T_R − T_ref)² at each step using minimize_scalar.
    Plant advances via true ODE (solve_ivp with CSTRMPCProblem._rhs).
    """
    from scipy.optimize import minimize_scalar

    # Resolve Q bounds
    try:
        from problems.cstr_mpc_problem import Q_RANGE
        q_lo, q_hi = float(Q_RANGE[0]), float(Q_RANGE[1])
    except ImportError:
        q_lo, q_hi = -8500.0, 0.0

    if opts.control_bounds is not None:
        if isinstance(opts.control_bounds, (tuple, list)) and len(opts.control_bounds) == 2:
            q_lo, q_hi = float(opts.control_bounds[0]), float(opts.control_bounds[1])
        elif isinstance(opts.control_bounds, dict):
            q_lo = float(opts.control_bounds.get("Q_lo", q_lo))
            q_hi = float(opts.control_bounds.get("Q_hi", q_hi))

    # Resolve target T_R
    t_ref = _resolve_scalar_target(opts.target, default=136.0,
                                    state_labels=problem.state_labels,
                                    target_key="T_R", target_idx=2)

    X_MIN = np.array([0.0, 0.0,  50.0,  50.0], dtype=np.float32)
    X_MAX = np.array([5.0, 5.0, 200.0, 200.0], dtype=np.float32)

    t_seg   = float(cfg.DT_SEG)
    n_steps = opts.n_steps

    xk    = x0.copy()
    x_log = [xk.copy()]
    Q_log = []
    J_log = []

    t_display_factor = float(getattr(problem, "time_factor", 60.0))
    t_arr = np.arange(n_steps + 1, dtype=np.float32) * t_seg * t_display_factor

    if opts.verbose:
        print(f"  T_ref={t_ref}°C  Q_bounds=[{q_lo:.0f}, {q_hi:.0f}] kJ/h")

    for step in range(n_steps):
        # Optimize Q using operator prediction
        res = minimize_scalar(
            fun    = lambda Q: float(
                (_predict_next_scalar(learner, xk, Q, X_MIN, X_MAX)[2] - t_ref) ** 2),
            bounds = (q_lo, q_hi),
            method = "bounded",
            options= {"xatol": 0.1, "maxiter": 500},
        )
        Q_opt = float(np.clip(res.x, q_lo, q_hi))

        # Advance TRUE ODE plant
        xk = _plant_step_scipy(problem, xk, Q_opt, t_seg, ref_solver="BDF")
        xk = np.clip(xk.astype(np.float32), X_MIN, X_MAX)

        x_log.append(xk.copy())
        Q_log.append(Q_opt)
        J_log.append(float(res.fun))

        if opts.verbose:
            print(f"  [step {step+1:3d}]  Q*={Q_opt:8.0f} kJ/h  "
                  f"T_R={float(xk[2]):.2f}°C  J={res.fun:.3e}")

    return (t_arr,
            np.stack(x_log, axis=0),
            np.array(Q_log, dtype=np.float32),
            np.array(J_log, dtype=np.float32))


# ── TripleTank tracking MPC ───────────────────────────────────────────

def _run_triple_tank_tracking_mpc(learner, problem, x0, opts, cfg):
    """2-D tracking MPC for TripleTank.

    Minimizes Σ(h_k − h_ref_k)² at each step using scipy minimize.
    """
    from scipy.optimize import minimize

    # Default control bounds for triple tank: Q1, Q2 in [0, 200] cm³/s
    try:
        from problems.triple_tank_mpc_problem import Q1_RANGE, Q2_RANGE
        q1_lo, q1_hi = float(Q1_RANGE[0]), float(Q1_RANGE[1])
        q2_lo, q2_hi = float(Q2_RANGE[0]), float(Q2_RANGE[1])
    except ImportError:
        q1_lo, q1_hi = 0.0, 200.0
        q2_lo, q2_hi = 0.0, 200.0

    if opts.control_bounds is not None:
        if isinstance(opts.control_bounds, dict):
            q1_lo = float(opts.control_bounds.get("Q1_lo", q1_lo))
            q1_hi = float(opts.control_bounds.get("Q1_hi", q1_hi))
            q2_lo = float(opts.control_bounds.get("Q2_lo", q2_lo))
            q2_hi = float(opts.control_bounds.get("Q2_hi", q2_hi))

    # Resolve target levels: h_ref for all 3 states
    h_ref = _resolve_vector_target(opts.target, n_states=3,
                                    default=[150.0, 150.0, 150.0],
                                    state_labels=problem.state_labels)

    X_MIN = np.zeros(3, dtype=np.float32)
    X_MAX = np.full(3, 350.0, dtype=np.float32)  # levels in cm (X0_UPPER=300)

    t_seg   = float(cfg.DT_SEG)
    n_steps = opts.n_steps

    xk    = x0.copy()
    x_log = [xk.copy()]
    u_log = []
    J_log = []

    t_display_factor = float(getattr(problem, "time_factor", 1.0))
    t_arr = np.arange(n_steps + 1, dtype=np.float32) * t_seg * t_display_factor
    bounds = [(q1_lo, q1_hi), (q2_lo, q2_hi)]
    h_ref_arr = np.asarray(h_ref, dtype=np.float32)

    if opts.verbose:
        print(f"  h_ref={h_ref_arr}  Q_bounds=[{q1_lo},{q1_hi}]×[{q2_lo},{q2_hi}]")

    u_prev = np.array([(q1_lo + q1_hi) / 2, (q2_lo + q2_hi) / 2], dtype=np.float64)
    for step in range(n_steps):
        def cost(u_vec):
            xn = _predict_next_vec(learner, xk, u_vec, X_MIN, X_MAX)
            return float(np.sum((xn - h_ref_arr) ** 2))

        res = minimize(cost, u_prev, method="SLSQP", bounds=bounds,
                       options={"ftol": 1e-4, "maxiter": 200})
        u_opt = np.clip(res.x, [q1_lo, q2_lo], [q1_hi, q2_hi]).astype(np.float32)

        # Advance true ODE plant
        xk = _plant_step_scipy_vec(problem, xk, u_opt, t_seg, ref_solver="RK45")
        xk = np.clip(xk.astype(np.float32), X_MIN, X_MAX)

        x_log.append(xk.copy())
        u_log.append(u_opt.copy())
        J_log.append(float(res.fun))
        u_prev = u_opt.astype(np.float64)

        if opts.verbose:
            print(f"  [step {step+1:3d}]  Q=[{u_opt[0]:.1f},{u_opt[1]:.1f}]  "
                  f"h={xk.tolist()}  J={res.fun:.3e}")

    return (t_arr,
            np.stack(x_log, axis=0),
            np.stack(u_log, axis=0),
            np.array(J_log, dtype=np.float32))


# ── Bioreactor economic MPC ───────────────────────────────────────────

def _run_bioreactor_empc(learner, problem, x0, opts, cfg):
    """Shrinking-horizon economic MPC for Fed-batch Bioreactor.

    Maximizes terminal Ps·Vs using SLSQP over a prediction horizon.
    Plant advances via true ODE (solve_ivp).
    """
    from scipy.optimize import minimize
    from scipy.integrate import solve_ivp

    INP_MIN  = 0.005
    INP_MAX  = 0.200

    if opts.control_bounds is not None:
        if isinstance(opts.control_bounds, (tuple, list)):
            INP_MIN, INP_MAX = float(opts.control_bounds[0]), float(opts.control_bounds[1])
        elif isinstance(opts.control_bounds, dict):
            INP_MIN = float(opts.control_bounds.get("inp_min", INP_MIN))
            INP_MAX = float(opts.control_bounds.get("inp_max", INP_MAX))

    YX_FIXED  = 0.40
    SIN_FIXED = 0.8
    V_MAX     = 5.0
    S_INH     = 0.8
    N_pred    = opts.horizon or 10

    X_MIN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    X_MAX = np.array([20.0, 5.0, 20.0, 15.0], dtype=np.float32)

    t_seg   = float(cfg.DT_SEG)
    n_steps = opts.n_steps

    t_display_factor = float(getattr(problem, "time_factor", 1.0))
    t_arr = np.arange(n_steps + 1, dtype=np.float32) * t_seg * t_display_factor

    xk       = x0.copy()
    x_log    = [xk.copy()]
    inp_log  = []
    J_log    = []
    inp_prev = 0.04

    def _op_rollout(xk, inp_seq, Yx, Sin):
        states = [xk.copy()]
        xc = xk.copy()
        for inp in inp_seq:
            z = np.array([xc[0], xc[1], xc[2], xc[3],
                          float(Yx), float(Sin), float(np.clip(inp, INP_MIN, INP_MAX))],
                         dtype=np.float32)[None, :]
            out = learner.predict_segment(z)
            xc  = np.clip(out["x_end"][0].astype(np.float32), X_MIN, X_MAX)
            states.append(xc.copy())
        return np.stack(states, axis=0)

    def _obj(inp_flat, xk_, inp_prev_, Yx_, Sin_):
        inp_seq = np.asarray(inp_flat, dtype=np.float64)
        states  = _op_rollout(xk_, inp_seq, Yx_, Sin_)
        J       = -float(states[-1, 2]) * float(states[-1, 3])  # -Ps*Vs
        J      -= 0.1 * float(np.sum(states[1:, 2] * states[1:, 3])) * t_seg
        J      += 0.01 * float(np.sum(inp_seq)) * t_seg
        diff    = np.concatenate([[inp_prev_], inp_seq])
        J      += 0.1 * float(np.sum((diff[1:] - diff[:-1]) ** 2))
        J      += 10.0 * float(np.sum(np.maximum(0.0, states[1:, 3] - V_MAX) ** 2))
        J      += 0.5  * float(np.sum(np.maximum(0.0, states[1:, 1] - S_INH) ** 2))
        J      += 10.0 * float(np.sum(np.maximum(0.0, -states[1:].astype(np.float64)) ** 2))
        return J

    def _plant_step_bio(xk_, inp_, Yx_, Sin_):
        theta = np.array([float(Yx_), float(Sin_), float(inp_)], dtype=np.float64)
        sol = solve_ivp(
            lambda t, y: problem.rhs_np(t, y, theta),
            t_span=(0.0, t_seg),
            y0=np.asarray(xk_, dtype=np.float64),
            method="RK45",
            rtol=1e-6, atol=1e-8,
        )
        return np.clip(sol.y[:, -1].astype(np.float32), X_MIN, X_MAX)

    if opts.verbose:
        print(f"  INP_MIN={INP_MIN}  INP_MAX={INP_MAX}  N_pred={N_pred}  V_max={V_MAX}")

    for step in range(n_steps):
        N_h = min(N_pred, n_steps - step)
        bnd = [(INP_MIN, INP_MAX)] * N_h
        u0  = np.full(N_h, inp_prev, dtype=np.float64)
        candidates = [u0, np.full(N_h, INP_MAX, dtype=np.float64)]

        best_res = None
        for cand in candidates:
            r = minimize(_obj, cand, args=(xk, inp_prev, YX_FIXED, SIN_FIXED),
                         method="SLSQP", bounds=bnd,
                         options={"ftol": 1e-4, "maxiter": 100})
            if best_res is None or r.fun < best_res.fun:
                best_res = r

        inp_apply = float(np.clip(best_res.x[0], INP_MIN, INP_MAX))

        # Advance TRUE ODE plant
        xk = _plant_step_bio(xk, inp_apply, YX_FIXED, SIN_FIXED)

        x_log.append(xk.copy())
        inp_log.append(inp_apply)
        J_log.append(float(best_res.fun))
        inp_prev = inp_apply

        if opts.verbose:
            PsVs = float(xk[2] * xk[3])
            print(f"  [step {step+1:3d}]  inp={inp_apply:.4f} L/min  "
                  f"Ps={xk[2]:.3f}  Vs={xk[3]:.3f}  Ps×Vs={PsVs:.3f} g")

    return (t_arr,
            np.stack(x_log, axis=0),
            np.array(inp_log, dtype=np.float32),
            np.array(J_log, dtype=np.float32))


# ── Target resolution helpers ─────────────────────────────────────────

def _resolve_scalar_target(target, default: float, state_labels,
                             target_key: str, target_idx: int) -> float:
    if target is None:
        return default
    if isinstance(target, dict):
        return float(target.get(target_key, default))
    if isinstance(target, (list, tuple)):
        for i, v in enumerate(target):
            if v is not None:
                return float(v)
        return default
    return float(target)


def _resolve_vector_target(target, n_states: int, default,
                            state_labels) -> list:
    if target is None:
        return list(default)
    if isinstance(target, dict):
        result = list(default)
        for i, lbl in enumerate(state_labels):
            if lbl in target:
                result[i] = float(target[lbl])
        return result
    if isinstance(target, (list, tuple)):
        result = []
        for i, v in enumerate(target):
            result.append(float(v) if v is not None else float(default[i]))
        return result
    return list(default)


# ── Metadata helpers ──────────────────────────────────────────────────

def _build_mpc_meta(system_name, mpc_legacy_name, opts, best_ckpt,
                     n_steps, elapsed):
    return {
        "system_name":         system_name,
        "mpc_problem_name":    mpc_legacy_name,
        "mode":                opts.mode,
        "basis":               opts.basis,
        "n_steps":             n_steps,
        "n_train":             opts.n_train,
        "epochs":              opts.epochs,
        "hidden":              opts.hidden,
        "n_layers":            opts.n_layers,
        "lr":                  opts.lr,
        "best_checkpoint":     best_ckpt,
        "elapsed_s":           elapsed,
        "ms_per_step":         (elapsed / n_steps * 1000) if n_steps > 0 else 0,
    }


def _write_meta(work_dir: str, metadata: dict) -> None:
    meta_path = os.path.join(work_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
             for k, v in metadata.items()},
            f, indent=2,
        )


def _copy_opts(opts: MPCOptions) -> MPCOptions:
    import copy
    return copy.copy(opts)


# ── Private path helper ────────────────────────────────────────────────

def _ensure_legacy_on_path() -> None:
    op_path = str(legacy_operator_root())
    if op_path not in sys.path:
        sys.path.insert(0, op_path)


# ── MPC operator inference helper ─────────────────────────────────────

def _run_mpc_operator_infer(result: "MPCResult", n_cases=4, cases=None,
                             reference=None, **kwargs) -> list:
    """Generate operator inference cases from an MPCResult.

    Loads the LPA surrogate (lpa_operator.npz) from checkpoint_dir and
    rolls it out on random cases sampled from state/control bounds.

    Returns list of dicts: [{"t", "y_op", "y_ref", "u", "x0", "case_id"}]
    """
    import warnings

    ckpt_dir = result.paths.get("checkpoint_dir")
    surr_path = os.path.join(ckpt_dir, "lpa_operator.npz") if ckpt_dir else None

    # Try generic LPA surrogate first
    if surr_path and os.path.exists(surr_path):
        return _infer_generic_surrogate(result, surr_path, n_cases, cases,
                                        reference, **kwargs)

    # Fall back to built-in learner (stored in operator_result or loaded fresh)
    learner = result.operator_result.get("learner") if result.operator_result else None
    if learner is not None:
        return _infer_builtin_learner(result, learner, n_cases, cases,
                                      reference, **kwargs)

    warnings.warn(
        "operator_inference_plot: no LPA surrogate found at "
        f"{surr_path!r} and no learner stored in operator_result. "
        "Run with train_operator=True to generate the surrogate.",
        stacklevel=3,
    )
    return []


def _infer_generic_surrogate(result, surr_path, n_cases, cases, reference,
                              **kwargs):
    """Roll out the LPA surrogate for n_cases random ICs."""
    from ..mpc._generic_mpc import _NumpyLPASurrogate
    from ..utils.reference import solve_reference_ivp

    surr = _NumpyLPASurrogate.load(surr_path)
    meta = result.metadata or {}
    dt   = float(meta.get("dt_seg", 0.02))
    n_seg = int(meta.get("n_seg", 25))

    sys = result.system
    state_b  = getattr(sys, "state_bounds",   {}) or {} if sys else {}
    ctrl_b   = getattr(sys, "control_bounds",  {}) or {} if sys else {}
    s_names  = list(getattr(sys, "state_names",   []) if sys else [])
    c_names  = list(getattr(sys, "control_names", []) if sys else [])

    s_lo = np.array([state_b.get(n, (-3.0, 3.0))[0] for n in s_names] or [-3.0])
    s_hi = np.array([state_b.get(n, (-3.0, 3.0))[1] for n in s_names] or [ 3.0])
    c_lo = np.array([ctrl_b.get(n, (-5.0, 5.0))[0] for n in c_names] or [-5.0])
    c_hi = np.array([ctrl_b.get(n, (-5.0, 5.0))[1] for n in c_names] or [ 5.0])

    if cases is None:
        rng = np.random.RandomState(77)
        cases = []
        for _ in range(n_cases):
            x0_i  = rng.uniform(s_lo, s_hi)
            u_seq = rng.uniform(c_lo, c_hi, (n_seg, len(c_lo)))
            cases.append({"x0": x0_i, "u_seq": u_seq})

    case_results = []
    for i, case in enumerate(cases):
        x0_c  = np.asarray(case["x0"], dtype=float)
        u_seq = np.asarray(case.get("u_seq",
                    np.zeros((n_seg, len(c_lo)), dtype=float)), dtype=float)

        states = [x0_c.copy()]
        xk = x0_c.copy()
        for step in range(n_seg):
            uk = u_seq[step].tolist() if step < len(u_seq) else [0.0] * len(c_lo)
            xk = surr.predict_next(xk, uk)
            states.append(xk.copy())
        y_op = np.stack(states).astype(float)
        t_c  = np.arange(n_seg + 1, dtype=float) * dt

        y_ref = None
        if reference == "solve_ivp" and sys is not None:
            try:
                # Build piecewise-constant control as first step
                u_first = u_seq[0].tolist() if len(u_seq) > 0 else None
                y_ref = solve_reference_ivp(sys, x0_c, t_c,
                                            controls=u_first, method="BDF")
            except Exception as exc:
                import warnings as _w
                _w.warn(f"solve_reference_ivp failed: {exc}", stacklevel=4)

        case_results.append({
            "t":       t_c,
            "y_op":    y_op,
            "y_ref":   y_ref,
            "u":       u_seq,
            "x0":      x0_c,
            "case_id": i,
        })

    return case_results


def _infer_builtin_learner(result, learner, n_cases, cases, reference,
                            **kwargs):
    """Roll out a built-in OperatorLearner for n_cases random ICs."""
    from ..workflows.operator_workflow import _rollout_np, _sample_cases

    meta  = result.metadata or {}
    n_seg = int(meta.get("n_seg", 25))
    dt    = float(meta.get("dt_seg", 0.02))

    # Build a minimal OperatorResult-like object for _sample_cases
    class _FakeOpResult:
        metadata = meta
        paths    = result.paths
        system   = result.system
        learner  = learner
        t        = None
        y        = None

    if cases is None:
        cases = _sample_cases(_FakeOpResult(), n_cases)

    case_results = []
    for i, case in enumerate(cases):
        x0_c = np.asarray(case["x0"], dtype=np.float32)
        p_c  = case.get("params")
        if p_c is not None:
            theta = np.asarray(p_c, dtype=np.float32).ravel()
        else:
            try:
                nom = learner.problem.nominal_input()
                theta = nom[learner.problem.state_dim:
                            learner.problem.state_dim + learner.problem.param_dim]
            except Exception:
                theta = np.zeros(1, dtype=np.float32)
        theta_seq = np.tile(theta, (n_seg, 1))
        y_op = _rollout_np(learner, x0_c, theta_seq).astype(float)
        t_c  = np.arange(n_seg + 1, dtype=float) * dt

        case_results.append({
            "t":       t_c,
            "y_op":    y_op,
            "y_ref":   None,
            "u":       None,
            "x0":      x0_c,
            "case_id": i,
        })

    return case_results
