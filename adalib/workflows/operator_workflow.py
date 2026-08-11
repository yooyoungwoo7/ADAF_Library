"""
adalib/workflows/operator_workflow.py
run_operator() — full operator-learning workflow.

Workflow steps executed internally:
  1. Resolve system name → legacy problem name
  2. Create run directory structure (data/, checkpoints/, logs/, results/)
  3. Set legacy environment variables; reload config-dependent modules
  4. Generate full-case trajectories (train + val)
  5. Slice into segment datasets; compute normalization statistics
  6. Save all generated data
  7. Build TensorFlow datasets
  8. Instantiate and train OperatorLearner
  9. Save training log; load best checkpoint
  10. Run rollout inference from provided x0 / params
  11. Save rollout result and metadata.json
  12. Return OperatorResult
"""
from __future__ import annotations

import os
import sys
import glob
import json
import importlib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..systems.registry import get_system
from ..systems.callable_system import CallableODESystem
from ..operator.options import OperatorOptions
from ..utils.legacy_context import reload_legacy_chain
from ..utils.paths import legacy_operator_root


# ── System → legacy problem name (for operator learning) ─────────────
_OPERATOR_PROBLEM_MAP: dict[str, str] = {
    "lotka_volterra":      "lotka",
    "fedbatch_bioreactor": "bioreactor",
    "cstr":                "cstr",
    "triple_tank":         "triple_tank",
}
_NO_OPERATOR_SUPPORT = {"euler"}


# ── Result dataclass ──────────────────────────────────────────────────
@dataclass
class OperatorResult:
    """Result returned by run_operator().

    Attributes
    ----------
    t : ndarray or None
        Segment-boundary time array of shape (N_SEG+1,) in hours.
        None when infer=False or no x0 provided.
    y : ndarray or None
        State array of shape (N_SEG+1, n_state) at segment boundaries.
        None when infer=False or no x0 provided.
    learner : OperatorLearner or None
        Trained learner instance.
    train_info : dict
        Dict returned by learner.fit() including best_checkpoint,
        best_val_phys_loss, or {"reused": True} when a checkpoint was reused.
    metadata : dict
        Configuration summary (system, basis, n_train, epochs, val loss, …).
    paths : dict
        Paths to all generated artifacts:
          work_dir, data_dir, segment_dir, stats_dir,
          checkpoint_dir, best_checkpoint, log_dir, result_dir.
    system : ODESystem or None
        The ODE system object (stored for reference='solve_ivp' support).
    """
    t:          Optional[np.ndarray]
    y:          Optional[np.ndarray]
    learner:    Any
    train_info: Dict
    metadata:   Dict
    paths:      Dict
    system:     Any = None   # optional: stored for solve_ivp reference

    # ── Post-processing convenience methods ───────────────────────────

    def plot(
        self,
        reference   = None,
        system      = None,
        x0          = None,
        params      = None,
        controls    = None,
        state_names = None,
        labels      = None,
        state_groups= None,
        save_path   = None,
        show: bool  = False,
        title       = None,
        **kwargs,
    ):
        """Plot the operator rollout stored in this result.

        Parameters
        ----------
        reference
            * ``"solve_ivp"`` — compare against a scipy BDF reference.
            * ``(t_ref, y_ref)`` tuple, scipy ``OdeResult``, or callable.
            * ``None`` — trajectory only.
        system
            Override the stored system for ``reference="solve_ivp"``.
        x0, params, controls
            Override the stored initial condition / parameters / controls.
        state_names, labels, state_groups
            Forwarded to :func:`adalib.utils.plot_operator_result`.
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes, metrics)
        """
        from ..utils.plotting import plot_operator_result
        _sys = system or self.system
        # If x0 not provided, use the stored first state (for reference simulation)
        _x0 = x0 if x0 is not None else (
            self.y[0].tolist() if self.y is not None else None
        )
        _ref = reference   # "solve_ivp" is handled internally by plot_operator_result

        return plot_operator_result(
            self,
            system      = _sys,
            x0          = _x0,
            params      = params,
            control     = controls,
            reference   = _ref,
            state_names = state_names,
            labels      = labels,
            state_groups= state_groups,
            save_path   = save_path,
            show        = show,
            title       = title,
            **kwargs,
        )

    def infer(
        self,
        x0       = None,
        params   = None,
        controls = None,
        cases    = None,
        n_cases: Optional[int] = None,
        n_steps: Optional[int] = None,
        save: bool = True,
        **kwargs,
    ):
        """Run operator inference on one or more cases.

        Parameters
        ----------
        x0, params, controls
            Single-case explicit IC / params / controls.
        cases
            List of dicts, each with keys ``"x0"``, ``"params"``,
            ``"controls"``.  When given, takes priority over ``n_cases``.
        n_cases
            Number of cases to sample automatically.  For built-in systems
            the validation full-cases data is used; for generic systems
            random (x0, u) pairs are sampled from the training distribution.
        n_steps
            Override the number of rollout steps (default: ``metadata["n_seg"]``).
        save
            If ``True``, save the inference result to
            ``paths["result_dir"]/inference.npz``.

        Returns
        -------
        list of dict
            Each dict has keys ``"t"``, ``"y_op"``, ``"y_ref"``,
            ``"u"``, ``"x0"``, ``"case_id"``.
        """
        return _run_infer(self, x0=x0, params=params, controls=controls,
                          cases=cases, n_cases=n_cases, n_steps=n_steps,
                          save=save, **kwargs)

    def inference_plot(
        self,
        cases       = None,
        n_cases: Optional[int] = 4,
        reference   = None,
        state_names = None,
        control_names = None,
        save_path   = None,
        show: bool  = False,
        title       = None,
        **kwargs,
    ):
        """Multi-panel operator inference validation plot.

        Generates ``n_cases`` rollouts, compares against ``reference``,
        and plots a ``(n_state + n_control) × n_cases`` grid.

        Parameters
        ----------
        cases
            Explicit list of case dicts (see :meth:`infer`).
        n_cases
            Number of cases to auto-generate when ``cases`` is None.
        reference
            * ``"solve_ivp"`` — compare each rollout against scipy.
            * Other forms forwarded to :meth:`infer`.
        state_names, control_names
            Axis labels.
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes)
        """
        from ..utils.plotting import plot_operator_inference
        _ref = reference if reference != "solve_ivp" else None
        do_ref = (reference == "solve_ivp")

        infer_cases = _run_infer(
            self, cases=cases, n_cases=n_cases,
            reference=reference if do_ref else None, **kwargs,
        )

        if state_names is None:
            state_names = self.metadata.get("state_names") or list(
                getattr(self.system, "state_names", []))
        sys_name = self.metadata.get("system_name", "")
        _title = title or f"{sys_name} — LPA Operator Inference Validation"

        return plot_operator_inference(
            infer_cases,
            state_names   = state_names,
            control_names = control_names,
            save_path     = save_path,
            show          = show,
            title         = _title,
        )

    def operator_infer(
        self,
        x0          = None,
        params      = None,
        controls    = None,
        n_cases: int = 4,
        reference   = None,
        state_names = None,
        control_names = None,
        save_path   = None,
        show: bool  = False,
        title       = None,
        **kwargs,
    ):
        """Inference + validation plot in one call — no boilerplate needed.

        Parameters
        ----------
        x0, params, controls
            Explicit IC / parameters / controls for a single case.
            When provided, a single-case rollout is plotted.
        n_cases
            Number of cases to auto-sample when x0/params/controls are omitted.
        reference
            ``"solve_ivp"`` to compare each rollout against scipy, or ``None``.
        state_names, control_names
            Axis labels (auto-resolved from system/metadata if omitted).
        save_path, show, title
            Standard plot options.

        Returns
        -------
        (fig, axes)
        """
        _cases = None
        if x0 is not None or params is not None or controls is not None:
            _cases = [{"x0": x0, "params": params, "controls": controls}]
        return self.inference_plot(
            cases         = _cases,
            n_cases       = n_cases,
            reference     = reference,
            state_names   = state_names,
            control_names = control_names,
            save_path     = save_path,
            show          = show,
            title         = title,
            **kwargs,
        )

    def load_inference(self, path: Optional[str] = None):
        """Load a previously saved inference .npz.

        Parameters
        ----------
        path
            Override the default ``paths["result_dir"]/inference.npz``.

        Returns
        -------
        dict of ndarray
        """
        from ..utils.artifacts import load_npz
        if path is None:
            result_dir = self.paths.get("result_dir", ".")
            path = os.path.join(result_dir, "inference.npz")
        return load_npz(path)

    def save_inference(self, data: Dict, path: Optional[str] = None) -> str:
        """Save inference data dict to a .npz file.

        Parameters
        ----------
        data
            Dict of array-like values.
        path
            Override the default ``paths["result_dir"]/inference.npz``.

        Returns
        -------
        str
            Path written to.
        """
        from ..utils.artifacts import save_npz
        if path is None:
            result_dir = self.paths.get("result_dir", ".")
            path = os.path.join(result_dir, "inference.npz")
        arrs = {k: np.asarray(v) for k, v in data.items()
                if v is not None}
        save_npz(path, **arrs)
        return path

    def list_artifacts(self):
        """List all files under ``paths['work_dir']``."""
        from ..utils.artifacts import list_run_artifacts
        work = self.paths.get("work_dir")
        return list_run_artifacts(work) if work else []


# ── Helpers ───────────────────────────────────────────────────────────

def _resolve_legacy_name(system) -> tuple[str, str]:
    """Return (system_name_str, legacy_problem_name) for a system arg."""
    if isinstance(system, str):
        system_name = system.lower()
        # Resolve aliases first
        from ..systems.registry import _ALIASES, _REGISTRY
        canonical = _ALIASES.get(system_name, system_name)
        system_name = canonical
    else:
        system_name = getattr(system, "name", type(system).__name__).lower()

    if isinstance(system, str):
        sys_obj = get_system(system)
    else:
        sys_obj = system

    if isinstance(sys_obj, CallableODESystem):
        raise NotImplementedError(
            "Generic operator learning for arbitrary CallableODESystem is not "
            "fully supported yet.  Use a built-in system such as 'cstr', "
            "'triple_tank', or 'fedbatch_bioreactor', or provide a custom "
            "OperatorProblem adapter."
        )

    for key, legacy in _OPERATOR_PROBLEM_MAP.items():
        if key in system_name or system_name in key:
            return system_name, legacy

    if any(k in system_name for k in _NO_OPERATOR_SUPPORT):
        raise NotImplementedError(
            f"Operator learning is not supported for system '{system_name}'. "
            "Supported: " + str(sorted(_OPERATOR_PROBLEM_MAP.keys()))
        )

    raise ValueError(
        f"Cannot map system '{system_name}' to a legacy operator problem. "
        "Supported: " + str(sorted(_OPERATOR_PROBLEM_MAP.keys()))
    )


def _build_dirs(opts: OperatorOptions, legacy_name: str) -> Dict[str, str]:
    work_dir   = opts.work_dir or f"./runs/{legacy_name}_operator"
    work_dir   = str(Path(work_dir).resolve())
    data_dir   = opts.data_dir or os.path.join(work_dir, "data")
    ckpt_dir   = opts.checkpoint_dir or os.path.join(work_dir, "checkpoints")
    log_dir    = opts.log_dir or os.path.join(work_dir, "logs")
    result_dir = opts.result_dir or os.path.join(work_dir, "results")

    for d in [data_dir, ckpt_dir, log_dir, result_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "work_dir":       work_dir,
        "data_dir":       data_dir,
        "segment_dir":    data_dir,
        "stats_dir":      data_dir,
        "checkpoint_dir": ckpt_dir,
        "best_checkpoint": None,
        "log_dir":        log_dir,
        "result_dir":     result_dir,
    }


# ── Public API ────────────────────────────────────────────────────────

def run_operator(
    system,
    x0=None,
    t_span=None,
    params=None,
    controls=None,
    options=None,
    **kwargs,
) -> OperatorResult:
    """Full operator-learning workflow.

    Parameters
    ----------
    system : str or ODESystem
        Built-in systems: ``"cstr"``, ``"triple_tank"``,
        ``"fedbatch_bioreactor"``, ``"lotka_volterra"``.
        Strings are resolved via :func:`get_system`.
    x0 : array-like, optional
        Initial state for rollout inference after training.
    t_span : (t0, t1), optional
        Used to verify integration range; ignored internally (N_SEG/DT_SEG
        come from the legacy problem config).
    params : array-like, optional
        ODE parameters for rollout.  Alias ``p`` accepted via kwargs.
        Replicated across all segments if 1-D.
    options : OperatorOptions or dict, optional
        Workflow options.  Any field can also be passed as a kwarg.
    **kwargs
        Override any :class:`OperatorOptions` field by name.

    Returns
    -------
    OperatorResult
    """
    # ── 0. CallableODESystem → generic operator path ──────────────────
    _sys_obj = get_system(system) if isinstance(system, str) else system
    if isinstance(_sys_obj, CallableODESystem):
        if options is None:
            opts = OperatorOptions()
        elif isinstance(options, dict):
            opts = OperatorOptions(**options)
        else:
            import copy as _copy
            opts = _copy.copy(options)
        for k, v in kwargs.items():
            if hasattr(opts, k):
                setattr(opts, k, v)
        if params is None:
            params = kwargs.get("p", None)
        return _run_generic_operator(_sys_obj, x0, t_span, params, opts)

    # ── 1. Resolve system ──────────────────────────────────────────────
    system_name, legacy_name = _resolve_legacy_name(system)

    # ── 2. Resolve options ─────────────────────────────────────────────
    if options is None:
        opts = OperatorOptions()
    elif isinstance(options, dict):
        opts = OperatorOptions(**options)
    else:
        import copy
        opts = copy.copy(options)

    for k, v in kwargs.items():
        if hasattr(opts, k):
            setattr(opts, k, v)

    if params is None:
        params = kwargs.get("p", None)

    # ── 3. Create work directories ─────────────────────────────────────
    paths = _build_dirs(opts, legacy_name)
    data_dir   = paths["data_dir"]
    ckpt_dir   = paths["checkpoint_dir"]
    log_dir    = paths["log_dir"]
    result_dir = paths["result_dir"]

    if opts.verbose:
        print(f"[run_operator] system={system_name}  legacy={legacy_name}  "
              f"basis={opts.basis}")
        print(f"[run_operator] work_dir={paths['work_dir']}")

    # ── 4. Configure legacy environment ───────────────────────────────
    _ensure_legacy_on_path()
    reload_legacy_chain(legacy_name, opts.basis)

    import config as _cfg
    import problems.registry as _preg
    import data.dataset_builder as _db
    import models.learner as _ml

    problem = _preg.get_problem(legacy_name)

    # ── 5. Data paths ──────────────────────────────────────────────────
    prefix = legacy_name
    train_full_path = os.path.join(data_dir, f"{prefix}_train_fullcases.npz")
    val_full_path   = os.path.join(data_dir, f"{prefix}_val_fullcases.npz")
    train_seg_path  = os.path.join(data_dir, f"{prefix}_train_segments.npz")
    val_seg_path    = os.path.join(data_dir, f"{prefix}_val_segments.npz")

    # ── 6. Data generation ─────────────────────────────────────────────
    need_data = opts.generate_data and not opts.reuse_existing_data
    if need_data or opts.force_rebuild_data:
        if opts.verbose:
            print(f"[run_operator] Generating {opts.n_train} train / "
                  f"{opts.n_val} val full-case trajectories...")
        _db.build_and_save_fullcase(problem, train_full_path, opts.n_train, seed=opts.seed)
        _db.build_and_save_fullcase(problem, val_full_path,   opts.n_val,   seed=opts.seed + 1)

        if opts.verbose:
            print("[run_operator] Slicing into segments...")
        train_seg = _db.build_segments_from_fullcase(
            problem, train_full_path, seed=opts.seed + 100)
        X_mean, X_std = _db.compute_input_stats(train_seg["X"])
        _db.save_segments(train_seg_path, train_seg, X_mean=X_mean, X_std=X_std)

        val_seg = _db.build_segments_from_fullcase(
            problem, val_full_path, seed=opts.seed + 101)
        _db.save_segments(val_seg_path, val_seg, X_mean=X_mean, X_std=X_std)

        if opts.verbose:
            print(f"[run_operator] Train segs: {train_seg['X'].shape[0]}  "
                  f"Val segs: {val_seg['X'].shape[0]}")

    elif opts.reuse_existing_data:
        for p_ in [train_full_path, val_full_path, train_seg_path, val_seg_path]:
            if not os.path.exists(p_):
                raise FileNotFoundError(
                    f"reuse_existing_data=True but required file is missing:\n"
                    f"  {p_}\n"
                    "Set reuse_existing_data=False (and generate_data=True) to regenerate."
                )
        if opts.verbose:
            print(f"[run_operator] Reusing existing data in {data_dir}")

    train = _db.load_segments(train_seg_path)
    val   = _db.load_segments(val_seg_path)

    # ── 7. Training ────────────────────────────────────────────────────
    best_ckpt_path = None
    train_info: Dict = {}
    learner = None

    if opts.train and not opts.reuse_existing_checkpoint:
        if opts.verbose:
            print(f"[run_operator] Training OperatorLearner  "
                  f"epochs={opts.epochs}  hidden={opts.hidden}  "
                  f"n_layers={opts.n_layers}  lr={opts.lr:.2e}")

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

        train_info = learner.fit(
            train_ds, val_ds,
            epochs         = opts.epochs,
            checkpoint_dir = ckpt_dir,
            use_lr_schedule= opts.use_lr_schedule,
        )
        best_ckpt_path = train_info.get("best_checkpoint")

        if best_ckpt_path:
            learner.load_weights(best_ckpt_path)
            paths["best_checkpoint"] = best_ckpt_path
            if opts.verbose:
                print(f"[run_operator] Loaded best checkpoint: {best_ckpt_path}")

        # Save training log
        if learner.history:
            log_path = os.path.join(log_dir, "training_log.npz")
            np.savez_compressed(
                log_path,
                **{k: np.array(v) for k, v in learner.history.items() if v},
            )
            if opts.verbose:
                best_val = train_info.get("best_val_phys_loss")
                print(f"[run_operator] Best val loss: {best_val}")
                print(f"[run_operator] Training log: {log_path}")

    elif opts.reuse_existing_checkpoint:
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*best*.weights.h5")))
        if not ckpt_files:
            raise FileNotFoundError(
                f"reuse_existing_checkpoint=True but no '*best*.weights.h5' "
                f"found in:\n  {ckpt_dir}\n"
                "Set reuse_existing_checkpoint=False to retrain."
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
        paths["best_checkpoint"] = best_ckpt_path
        train_info = {"best_checkpoint": best_ckpt_path, "reused": True}

        if opts.verbose:
            print(f"[run_operator] Reused checkpoint: {best_ckpt_path}")

    # ── 8. Rollout inference ───────────────────────────────────────────
    t_out: Optional[np.ndarray] = None
    y_out: Optional[np.ndarray] = None

    if opts.infer and x0 is not None and learner is not None:
        n_seg  = _cfg.N_SEG
        dt_seg = _cfg.DT_SEG

        x0_arr = np.asarray(x0, dtype=np.float32)

        if params is not None:
            theta_arr = np.asarray(params, dtype=np.float32)
            if theta_arr.ndim == 1:
                theta_seq = np.tile(theta_arr, (n_seg, 1))
            else:
                theta_seq = theta_arr
        else:
            nom = problem.nominal_input()
            theta_flat = nom[problem.state_dim : problem.state_dim + problem.param_dim]
            theta_seq = np.tile(theta_flat, (n_seg, 1))

        if opts.verbose:
            print(f"[run_operator] Rollout: {n_seg} segments, "
                  f"T={n_seg * dt_seg:.3f} h")

        states = _rollout_np(learner, x0_arr, theta_seq)
        t_out  = np.arange(n_seg + 1, dtype=np.float32) * dt_seg
        y_out  = states

        rollout_path = os.path.join(result_dir, "rollout.npz")
        np.savez_compressed(rollout_path, t=t_out, y=y_out, x0=x0_arr)
        if opts.verbose:
            print(f"[run_operator] Rollout saved: {rollout_path}")

    # ── 9. Metadata ────────────────────────────────────────────────────
    metadata = {
        "system_name":         system_name,
        "legacy_problem_name": legacy_name,
        "basis":               opts.basis,
        "n_train":             opts.n_train,
        "n_val":               opts.n_val,
        "epochs":              opts.epochs,
        "hidden":              opts.hidden,
        "n_layers":            opts.n_layers,
        "lr":                  opts.lr,
        "best_val_loss":       train_info.get("best_val_phys_loss"),
        "best_checkpoint":     best_ckpt_path,
        "n_seg":               int(_cfg.N_SEG),
        "dt_seg":              float(_cfg.DT_SEG),
        "t_final":             float(_cfg.T_FINAL),
    }

    meta_path = os.path.join(paths["work_dir"], "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in metadata.items()},
            f, indent=2,
        )

    if opts.verbose:
        print(f"[run_operator] Metadata: {meta_path}")

    _sys_final = get_system(system_name) if isinstance(system, str) else system
    return OperatorResult(
        t=t_out,
        y=y_out,
        learner=learner,
        train_info=train_info,
        metadata=metadata,
        paths=paths,
        system=_sys_final,
    )


# ── Private helpers ────────────────────────────────────────────────────

def _ensure_legacy_on_path() -> None:
    op_path = str(legacy_operator_root())
    if op_path not in sys.path:
        sys.path.insert(0, op_path)


def _run_generic_operator(system, x0, t_span, params, opts: OperatorOptions) -> OperatorResult:
    """Generic operator learning for a CallableODESystem via LPA NN.

    Physics-residual only — matches the philosophy of the built-in
    (cstr/triple_tank/fedbatch_bioreactor/lotka_volterra) Operator pipeline.
    No reference trajectory is generated or fit against; the LPA basis's
    own analytic derivative is trained to satisfy system.rhs() directly.
    See adalib/mpc/_generic_mpc.py's _sample_inputs/_train_lpa_operator_physics.
    """
    import os, json
    import numpy as np
    from ..mpc._generic_mpc import _sample_inputs, _train_lpa_operator_physics, _NumpyLPASurrogate

    system_name = system.name

    # ── Dirs ──────────────────────────────────────────────────────────
    paths = _build_dirs(opts, system_name)
    data_dir = paths["data_dir"]
    ckpt_dir = paths["checkpoint_dir"]

    # ── dt and n_seg ──────────────────────────────────────────────────
    n_seg = int(getattr(opts, "n_seg", 25))
    dt    = getattr(opts, "dt", None)
    if dt is None and t_span is not None:
        dt = (float(t_span[1]) - float(t_span[0])) / n_seg
    if dt is None or float(dt) <= 0:
        raise ValueError(
            "opts.dt must be a positive float (segment duration). "
            "Alternatively supply t_span so dt = (t1-t0)/n_seg."
        )
    dt = float(dt)

    n_state   = len(system.state_names)
    ctrl_names = list(system.control_names or [])
    n_control  = len(ctrl_names)

    # ── Bounds ────────────────────────────────────────────────────────
    state_b = system.state_bounds or {}
    ctrl_b  = system.control_bounds or {}
    state_lo = np.array([state_b.get(n, (-10.0, 10.0))[0] for n in system.state_names])
    state_hi = np.array([state_b.get(n, (-10.0, 10.0))[1] for n in system.state_names])
    if ctrl_names:
        ctrl_lo = np.array([ctrl_b.get(n, (-1e6, 1e6))[0] for n in ctrl_names])
        ctrl_hi = np.array([ctrl_b.get(n, (-1e6, 1e6))[1] for n in ctrl_names])
    else:
        ctrl_lo = ctrl_hi = np.zeros(1)

    # ── LPA hyperparameters ───────────────────────────────────────────
    _LPA_N_P = 8; _LPA_MAX_ORDER = 6; _LPA_NT_SEG = 20
    N_p       = int(getattr(opts, "lpa_n_panels",  _LPA_N_P))
    max_order = int(getattr(opts, "lpa_max_order", _LPA_MAX_ORDER))
    Nt        = int(getattr(opts, "lpa_nt_seg",    _LPA_NT_SEG))

    surr_path = os.path.join(ckpt_dir, "lpa_operator.npz")
    data_path = os.path.join(data_dir, "generic_data.npz")

    # ── Input sampling (physics-only: no reference trajectory needed) ──
    X_input_tr = X_input_va = None
    if opts.generate_data and not opts.reuse_existing_data:
        if opts.verbose:
            print(f"[run_operator/generic] Sampling {opts.n_train} train / "
                  f"{opts.n_val} val (x0,u) inputs  dt={dt:.3e} s  Nt={Nt}  "
                  f"(physics-only — no solve_ivp reference generated)")
        X_input_tr, X_input_va = _sample_inputs(
            opts.n_train, opts.n_val, n_state, n_control,
            state_lo, state_hi, ctrl_lo, ctrl_hi, seed=opts.seed,
        )
        np.savez_compressed(data_path,
                            X_input_tr=X_input_tr, X_input_va=X_input_va)
        if opts.verbose:
            print(f"[run_operator/generic] Inputs saved → {data_path}")
    elif opts.reuse_existing_data:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"reuse_existing_data=True but not found:\n  {data_path}"
            )
        d = np.load(data_path)
        X_input_tr, X_input_va = d["X_input_tr"], d["X_input_va"]
        if opts.verbose:
            print(f"[run_operator/generic] Loaded inputs: "
                  f"{X_input_tr.shape[0]} train samples")

    # ── Training (physics-residual only) ────────────────────────────────
    surrogate = None
    train_info: dict = {}

    if opts.train and not opts.reuse_existing_checkpoint:
        if X_input_tr is None:
            raise RuntimeError(
                "train=True but no inputs available. "
                "Set generate_data=True or reuse_existing_data=True."
            )
        if opts.verbose:
            print(f"[run_operator/generic] Training LPA Operator NN "
                  f"(physics-residual)  epochs={opts.epochs}  "
                  f"hidden={opts.hidden}  n_layers={opts.n_layers}  N_p={N_p}")
        surrogate = _train_lpa_operator_physics(
            X_input_tr, X_input_va, system,
            n_state, n_control, dt, opts, state_lo, state_hi,
        )
        surrogate.save(surr_path)
        train_info = {"best_checkpoint": surr_path}
        if opts.verbose:
            print(f"[run_operator/generic] LPA operator saved → {surr_path}")

    elif opts.reuse_existing_checkpoint:
        if not os.path.exists(surr_path):
            raise FileNotFoundError(
                f"reuse_existing_checkpoint=True but not found:\n  {surr_path}"
            )
        surrogate = _NumpyLPASurrogate.load(surr_path)
        train_info = {"best_checkpoint": surr_path, "reused": True}
        if opts.verbose:
            print(f"[run_operator/generic] Loaded LPA operator ← {surr_path}")

    # ── Rollout inference ─────────────────────────────────────────────
    t_out: np.ndarray | None = None
    y_out: np.ndarray | None = None

    if opts.infer and x0 is not None and surrogate is not None:
        x0_arr  = np.asarray(x0, dtype=np.float64)
        u_const = (np.asarray(params, dtype=np.float64).ravel()[:n_control]
                   if params is not None else np.zeros(n_control))

        states = [x0_arr.copy()]
        xk = x0_arr.copy()
        for _ in range(n_seg):
            xk = surrogate.predict_next(xk, u_const.tolist())
            states.append(xk.copy())
        y_out = np.stack(states).astype(np.float32)   # (n_seg+1, n_state)
        t_out = (np.arange(n_seg + 1, dtype=np.float32) * dt)

        rollout_path = os.path.join(paths["result_dir"], "rollout.npz")
        np.savez_compressed(rollout_path, t=t_out, y=y_out, x0=x0_arr)
        paths["best_checkpoint"] = surr_path
        if opts.verbose:
            print(f"[run_operator/generic] Rollout ({n_seg} segments) saved → {rollout_path}")

    # ── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "system_name":  system_name,
        "basis":        "lpa",
        "training":     "physics",
        "n_train":      opts.n_train,
        "n_val":        opts.n_val,
        "epochs":       opts.epochs,
        "hidden":       opts.hidden,
        "n_layers":     opts.n_layers,
        "lr":           opts.lr,
        "lpa_n_panels": N_p,
        "lpa_max_order": max_order,
        "lpa_nt_seg":   Nt,
        "n_seg":        n_seg,
        "dt_seg":       dt,
        "t_final":      float(n_seg * dt),
    }
    meta_path = os.path.join(paths["work_dir"], "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in metadata.items()},
            f, indent=2,
        )

    return OperatorResult(
        t=t_out,
        y=y_out,
        learner=None,
        train_info=train_info,
        metadata=metadata,
        paths=paths,
        system=system,
    )


def _rollout_np(learner, x0: np.ndarray, theta_seq: np.ndarray) -> np.ndarray:
    """Inference-only multi-step rollout without importing adalib.operator."""
    import numpy as np
    states = [x0.copy()]
    xk = x0.copy()
    for theta in theta_seq:
        z = np.concatenate([xk, theta], axis=0).astype(np.float32)[None, :]
        out = learner.predict_segment(z)
        xk = out["x_end"][0].astype(np.float32)
        states.append(xk)
    return np.stack(states, axis=0)


# ── Inference helper ────────────────────────────────────────────────────

def _run_infer(result: "OperatorResult",
               x0=None, params=None, controls=None,
               cases=None, n_cases=None, n_steps=None,
               reference=None, save: bool = False, **kwargs) -> list:
    """Generate inference case dicts from an OperatorResult.

    Returns list of dicts: [{"t", "y_op", "y_ref", "u", "x0", "case_id"}]
    """
    # Build explicit case list
    if cases is None and n_cases is not None:
        cases = _sample_cases(result, n_cases, n_steps=n_steps)
    elif cases is None:
        # Single case: use stored result or provided x0
        if x0 is not None:
            cases = [{"x0": x0, "params": params, "controls": controls}]
        elif result.t is not None and result.y is not None:
            # Return the already-computed rollout
            return [{"t":      np.asarray(result.t, dtype=float),
                     "y_op":   np.asarray(result.y, dtype=float),
                     "y_ref":  None, "u": None,
                     "x0":     None, "case_id": 0}]
        else:
            raise ValueError(
                "No inference data available. "
                "Provide x0, cases, or n_cases."
            )

    case_results = []
    for i, case in enumerate(cases):
        x0_c  = np.asarray(case.get("x0"), dtype=float)
        p_c   = case.get("params")
        u_c   = case.get("controls")
        u_seq = case.get("u_seq")  # generic: pre-built control sequence

        t_c, y_op_c = _rollout_one_case(result, x0_c, p_c, u_c, u_seq, n_steps)

        y_ref_c = None
        if reference == "solve_ivp":
            y_ref_c = _reference_one_case(result, x0_c, p_c, u_c, u_seq, t_c)

        case_results.append({
            "t":       t_c,
            "y_op":    y_op_c,
            "y_ref":   y_ref_c,
            "u":       u_seq if u_seq is not None else u_c,
            "x0":      x0_c,
            "case_id": i,
        })

    if save:
        _save_infer_cases(result, case_results)

    return case_results


def _rollout_one_case(result, x0, params, controls, u_seq, n_steps):
    """Run one operator rollout.  Handles built-in (learner) and generic (surrogate)."""
    import numpy as np

    meta = result.metadata or {}
    n_seg = n_steps or int(meta.get("n_seg", 25))
    dt    = float(meta.get("dt_seg", 0.02))

    # ── generic (LPA surrogate, no learner) ──────────────────────────
    if result.learner is None:
        surr = _load_generic_surrogate(result)
        if surr is None:
            raise RuntimeError(
                "No LPA surrogate found. "
                "Ensure the operator checkpoint exists in paths['checkpoint_dir']."
            )
        n_control = surr.n_state  # rough check; use u_seq if provided
        if u_seq is None:
            rng   = np.random.RandomState(42)
            n_ctl = len(meta.get("control_names") or
                        getattr(result.system, "control_names", None) or [1])
            if isinstance(n_ctl, list):
                n_ctl = len(n_ctl)
            u_seq = rng.uniform(-5.0, 5.0, (n_seg, n_ctl if n_ctl > 0 else 1))
        elif isinstance(u_seq, np.ndarray) and u_seq.ndim == 1:
            u_seq = u_seq[:, np.newaxis]

        states = [x0.copy()]
        xk = x0.copy()
        for step in range(n_seg):
            uk = u_seq[step].tolist() if step < len(u_seq) else [0.0]
            xk = surr.predict_next(xk, uk)
            states.append(xk.copy())
        y_op = np.stack(states).astype(float)
        t_c  = np.arange(n_seg + 1, dtype=float) * dt
        return t_c, y_op

    # ── built-in (learner available) ─────────────────────────────────
    x0_arr = x0.astype(np.float32)
    if params is not None:
        theta_flat = np.asarray(params, dtype=np.float32).ravel()
    else:
        # Use the learner's problem nominal if available
        try:
            nom = result.learner.problem.nominal_input()
            theta_flat = nom[result.learner.problem.state_dim:
                             result.learner.problem.state_dim
                             + result.learner.problem.param_dim]
        except Exception:
            theta_flat = np.zeros(1, dtype=np.float32)

    theta_seq = np.tile(theta_flat, (n_seg, 1))
    y_op = _rollout_np(result.learner, x0_arr, theta_seq).astype(float)
    t_c  = np.arange(n_seg + 1, dtype=float) * dt
    return t_c, y_op


def _reference_one_case(result, x0, params, controls, u_seq, t_c):
    """Compute solve_ivp reference for one case."""
    import warnings
    from ..utils.reference import solve_reference_ivp, _broadcast_control

    sys = result.system
    if sys is None:
        warnings.warn(
            "reference='solve_ivp' requires system to be stored. "
            "Pass system=... to inference_plot().",
            stacklevel=4,
        )
        return None

    # For generic systems: use first control step if u_seq provided
    u = None
    if u_seq is not None:
        u_arr = np.asarray(u_seq, dtype=float)
        u = u_arr[0].tolist() if u_arr.ndim == 2 else [float(u_arr[0])]
    elif controls is not None:
        u = _broadcast_control(controls)

    try:
        y_ref = solve_reference_ivp(sys, x0, t_c, params=params, controls=u,
                                    method="BDF")
        return y_ref
    except Exception as exc:
        warnings.warn(f"solve_reference_ivp failed: {exc}", stacklevel=4)
        return None


def _sample_cases(result, n_cases, n_steps=None):
    """Sample n_cases cases from the training distribution."""
    import numpy as np

    meta  = result.metadata or {}
    n_seg = n_steps or int(meta.get("n_seg", 25))
    rng   = np.random.RandomState(99)

    # ── generic: use state / control bounds from system ──────────────
    if result.learner is None:
        sys = result.system
        state_b   = getattr(sys, "state_bounds",   {}) or {} if sys else {}
        ctrl_b    = getattr(sys, "control_bounds",  {}) or {} if sys else {}
        s_names   = list(getattr(sys, "state_names",   []) if sys else meta.get("state_names", []))
        c_names   = list(getattr(sys, "control_names", []) if sys else [])

        s_lo = np.array([state_b.get(n, (-3.0, 3.0))[0] for n in s_names] or [-3.0])
        s_hi = np.array([state_b.get(n, (-3.0, 3.0))[1] for n in s_names] or [ 3.0])
        c_lo = np.array([ctrl_b.get(n, (-5.0, 5.0))[0] for n in c_names] or [-5.0])
        c_hi = np.array([ctrl_b.get(n, (-5.0, 5.0))[1] for n in c_names] or [ 5.0])

        cases = []
        for _ in range(n_cases):
            x0_i = rng.uniform(s_lo, s_hi)
            u_seq = rng.uniform(c_lo, c_hi, (n_seg, len(c_lo)))
            cases.append({"x0": x0_i, "params": None, "controls": None,
                          "u_seq": u_seq})
        return cases

    # ── built-in: try to load val full-cases ─────────────────────────
    data_dir   = result.paths.get("data_dir", ".")
    sys_name   = meta.get("legacy_problem_name", meta.get("system_name", ""))
    prefix     = sys_name
    val_path   = os.path.join(data_dir, f"{prefix}_val_fullcases.npz")

    if os.path.exists(val_path):
        d = np.load(val_path, allow_pickle=True)
        # Standard keys: "x0_cases" (N, n_state), "params_cases" (N, n_param)
        try:
            all_x0 = d["x0_cases"]
            all_p  = d.get("params_cases")
            idx    = rng.choice(len(all_x0), size=min(n_cases, len(all_x0)),
                                replace=False)
            cases  = []
            for i in idx:
                p_i = all_p[i].tolist() if all_p is not None else None
                cases.append({"x0": all_x0[i].tolist(), "params": p_i,
                               "controls": None})
            return cases
        except Exception:
            pass

    # Fallback: use the stored result x0 repeated with small perturbations
    if result.t is not None and result.y is not None:
        x0_base = np.asarray(result.y[0], dtype=float)
        cases = []
        for _ in range(n_cases):
            noise = rng.normal(0, 0.05 * np.abs(x0_base).clip(1e-3), x0_base.shape)
            cases.append({"x0": (x0_base + noise).tolist(),
                          "params": None, "controls": None})
        return cases

    raise RuntimeError(
        f"Cannot auto-generate {n_cases} cases: no val data found at {val_path!r} "
        "and no stored rollout. Pass explicit cases or run with generate_data=True."
    )


def _load_generic_surrogate(result):
    """Load _NumpyLPASurrogate from checkpoint if available."""
    ckpt_dir = result.paths.get("checkpoint_dir")
    if not ckpt_dir:
        return None
    surr_path = os.path.join(ckpt_dir, "lpa_operator.npz")
    if not os.path.exists(surr_path):
        return None
    from ..mpc._generic_mpc import _NumpyLPASurrogate
    return _NumpyLPASurrogate.load(surr_path)


def _save_infer_cases(result, case_results):
    """Save inference cases to result_dir/inference.npz."""
    result_dir = result.paths.get("result_dir", ".")
    path = os.path.join(result_dir, "inference.npz")
    try:
        arrs = {}
        for i, c in enumerate(case_results):
            arrs[f"t_{i}"]     = c["t"]
            arrs[f"y_op_{i}"]  = c["y_op"]
            if c.get("y_ref") is not None:
                arrs[f"y_ref_{i}"] = c["y_ref"]
        np.savez_compressed(path, **arrs)
    except Exception:
        pass
