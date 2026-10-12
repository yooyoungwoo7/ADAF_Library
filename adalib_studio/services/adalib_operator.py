"""
Operator learning via adalib.run_operator (runs inside worker processes only).

Training and inference are separate jobs. A training job leaves a model
directory (adalib's own work_dir layout) plus ``model.json``, a manifest with
everything inference needs: source, system, training domain, time grid and the
network size. Inference re-enters ``run_operator`` with
``train=False, reuse_existing_checkpoint=True``, which is adalib's documented
reload path; nothing is retrained.

Two adalib paths (see docs/adalib_operator_inverse_mpc.md §1):
  custom  -> CallableODESystem, generic physics-residual-only LPA operator.
             Inputs are the control values u, held constant over the rollout.
  builtin -> legacy problems (cstr, triple_tank, fedbatch_bioreactor) with
             their own time grid, sampling ranges and θ vector.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schemas.common import SystemSpec
from ..schemas.operator import OperatorInferRequest, OperatorTrainRequest
from .result_serializer import finite_list, write_json
from .system_builder import build_operator_system

MANIFEST = "model.json"
FINE_POINTS = 401


def _operator_options(**fields):
    from adalib import OperatorOptions
    return OperatorOptions(**fields)


# ── training ───────────────────────────────────────────────────────────

def run_operator_train_job(req: OperatorTrainRequest, job_dir: Path) -> Dict:
    import adalib
    from adalib import run_operator

    work = (job_dir / "model").resolve()
    o = req.options
    common = dict(n_train=o.n_train, n_val=o.n_val, epochs=o.epochs, batch_size=o.batch_size, lr=o.lr,
                  hidden=o.hidden, n_layers=o.n_layers, seed=o.seed, work_dir=str(work), verbose=True)
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if req.source == "custom":
            spec = req.system
            system = build_operator_system(spec)
            x0 = [float(s.initial) for s in spec.states]
            u = [float(c.value) for c in spec.controls]
            dt = req.t_final / req.n_seg
            print(f"[studio] adalib {adalib.__version__}: run_operator(CallableODESystem {system.name!r}, "
                  f"generic physics-residual path, n_seg={req.n_seg}, dt={dt:g})", flush=True)
            opts = _operator_options(**common, n_seg=req.n_seg, dt=dt, lpa_n_panels=o.lpa_n_panels,
                                     lpa_max_order=o.lpa_max_order, lpa_nt_seg=o.lpa_nt_seg)
            result = run_operator(system, x0=x0, t_span=(0.0, req.t_final), params=u or None, options=opts)
            train_sec = time.perf_counter() - t_start
            manifest = _custom_manifest(req, spec, result, work)
            history = None   # generic path keeps no history object; the API parses the log
        else:
            print(f"[studio] adalib {adalib.__version__}: run_operator({req.builtin!r}, built-in legacy problem)",
                  flush=True)
            opts = _operator_options(**common, basis=o.basis, use_lr_schedule=o.use_lr_schedule)
            result = run_operator(req.builtin, x0=None, options=opts)
            train_sec = time.perf_counter() - t_start
            manifest = _builtin_manifest(req, result, work)
            h = getattr(result.learner, "history", None) or {}
            history = {k: finite_list(v) for k, v in h.items()}
    manifest.update(adalib_version=adalib.__version__, options=o.model_dump(), train_seconds=train_sec,
                    trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"), label=req.label or manifest["system_name"])
    write_json(job_dir / MANIFEST, manifest)

    # preview: one prediction at the nominal inputs, through the same reload path inference uses
    print("[studio] preview prediction at the nominal inputs (reloaded checkpoint)", flush=True)
    preview = predict(manifest, job_dir, manifest["nominal"]["x0"], manifest["nominal"]["inputs"], reference=True)

    return {
        "kind": "operator_train",
        "adalib_version": adalib.__version__,
        "model_id": job_dir.name,
        "manifest": manifest,
        "train_info": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in (result.train_info or {}).items()},
        "history": history,
        "preview": preview,
        "train_seconds": train_sec,
        "warnings": _warnings(caught),
    }


def _custom_manifest(req: OperatorTrainRequest, spec: SystemSpec, result, work: Path) -> Dict:
    return {
        "model_id": work.parent.name,
        "source": "custom",
        "system_name": spec.name,
        "spec": spec.model_dump(),
        "state_names": spec.state_names,
        "input_names": spec.control_names,
        "input_kind": "control",
        "domain": {"x0": [[s.lower, s.upper] for s in spec.states],
                   "inputs": [[c.lower, c.upper] for c in spec.controls]},
        "nominal": {"x0": [s.initial for s in spec.states], "inputs": [c.value for c in spec.controls]},
        "n_seg": req.n_seg,
        "dt_seg": req.t_final / req.n_seg,
        "t_final": req.t_final,
        "time_unit": "",
        "time_factor": 1.0,
        "work_dir": "model",
        "best_val_loss": None,
        "path": "generic physics-residual-only LPA operator (CallableODESystem)",
    }


# Legacy problem attributes holding the sampling ranges (cstr_mpc_op/problems/*.py).
_BUILTIN_RANGES = {
    "cstr": (None, ["ALPHA_RANGE", "BETA_RANGE", "F_RANGE", "Q_RANGE"]),
    "triple_tank": (["H1_RANGE", "H2_RANGE", "H3_RANGE"], ["Q1_RANGE", "Q2_RANGE"]),
    "fedbatch_bioreactor": (["XS0_RANGE", "SS0_RANGE", "PS0_RANGE", "VS0_RANGE"],
                            ["YX_RANGE", "SIN_RANGE", "INP_RANGE"]),
}


def _builtin_manifest(req: OperatorTrainRequest, result, work: Path) -> Dict:
    P = result.learner.problem
    x_attrs, th_attrs = _BUILTIN_RANGES[req.builtin]
    if x_attrs is None:
        # CSTR samples x0 only within ±X0_PERTURB_FRAC·(upper−lower) of X0_NOMINAL (cstr_problem.sample_cases)
        span = P.X0_PERTURB_FRAC * (P.X0_UPPER - P.X0_LOWER)
        lo = np.maximum(P.X0_LOWER, P.X0_NOMINAL - span)
        hi = np.minimum(P.X0_UPPER, P.X0_NOMINAL + span)
        x_dom = [[float(a), float(b)] for a, b in zip(lo, hi)]
    else:
        x_dom = [[float(v) for v in getattr(P, a)] for a in x_attrs]
    th_dom = [[float(v) for v in getattr(P, a)] for a in th_attrs]
    nom = np.asarray(P.nominal_input(), dtype=float)
    meta = result.metadata
    return {
        "model_id": work.parent.name,
        "source": "builtin",
        "builtin": req.builtin,
        "system_name": req.builtin,
        "legacy_problem": meta.get("legacy_problem_name"),
        "state_names": list(P.state_labels),
        "input_names": list(P.param_labels),
        "input_kind": "parameter",
        "domain": {"x0": x_dom, "inputs": th_dom},
        "nominal": {"x0": nom[:P.state_dim].tolist(), "inputs": nom[P.state_dim:P.state_dim + P.param_dim].tolist()},
        "n_seg": int(meta["n_seg"]),
        "dt_seg": float(meta["dt_seg"]),
        "t_final": float(meta["t_final"]),
        "time_unit": P.time_unit,
        "time_factor": float(getattr(P, "time_factor", 1.0)),
        "work_dir": "model",
        "best_val_loss": meta.get("best_val_loss"),
        "reference_method": f"{P.ref_solver} (rtol={P.ref_rtol:g}, atol={P.ref_atol:g})",
        "path": f"built-in legacy problem '{meta.get('legacy_problem_name')}' (data + physics trained)",
    }


# ── inference ──────────────────────────────────────────────────────────

def run_operator_infer_job(req: OperatorInferRequest, job_dir: Path, runs_dir: Path) -> Dict:
    import adalib

    model_dir = (runs_dir / req.model_id).resolve()
    manifest = json.loads((model_dir / MANIFEST).read_text(encoding="utf-8"))
    print(f"[studio] adalib {adalib.__version__}: operator inference with model {req.model_id} "
          f"({manifest['source']}: {manifest['system_name']})", flush=True)
    t0 = time.perf_counter()
    out = predict(manifest, model_dir, req.x0, req.inputs, reference=req.reference, result_dir=job_dir / "rollout")
    out.update(kind="operator_infer", adalib_version=adalib.__version__, model_id=req.model_id,
               manifest=manifest, wall_seconds=time.perf_counter() - t0)
    return out


def predict(manifest: Dict, model_dir: Path, x0: List[float], inputs: List[float], reference: bool,
            result_dir: Optional[Path] = None) -> Dict:
    """One rollout through adalib's reload path, plus an optional reference."""
    from adalib import run_operator

    n_state, n_in = len(manifest["state_names"]), len(manifest["input_names"])
    if len(x0) != n_state or len(inputs) != n_in:
        raise ValueError(f"expected {n_state} initial values and {n_in} inputs, got {len(x0)} and {len(inputs)}")
    work = (model_dir / manifest["work_dir"]).resolve()
    o = manifest["options"]
    # checkpoint/data come from the model; anything the call writes goes to result_dir
    out_dir = (result_dir or work / "preview").resolve()
    dirs = dict(work_dir=str(out_dir), checkpoint_dir=str(work / "checkpoints"), data_dir=str(work / "data"),
                result_dir=str(out_dir / "results"), log_dir=str(out_dir / "logs"))
    reuse = dict(train=False, reuse_existing_checkpoint=True, hidden=o["hidden"], n_layers=o["n_layers"],
                 verbose=False)
    warn_domain = _domain_warnings(manifest, x0, inputs)

    t0 = time.perf_counter()
    if manifest["source"] == "custom":
        spec = SystemSpec.model_validate(manifest["spec"])
        system = build_operator_system(spec)
        opts = _operator_options(**dirs, **reuse, generate_data=False, reuse_existing_data=False,
                                 n_seg=manifest["n_seg"], dt=manifest["dt_seg"], lpa_n_panels=o["lpa_n_panels"],
                                 lpa_max_order=o["lpa_max_order"], lpa_nt_seg=o["lpa_nt_seg"])
        r = run_operator(system, x0=list(x0), t_span=(0.0, manifest["t_final"]), params=list(inputs) or None,
                         options=opts)
    else:
        opts = _operator_options(**dirs, **reuse, generate_data=False, reuse_existing_data=True,
                                 basis=o["basis"])
        r = run_operator(manifest["builtin"], x0=list(x0), params=list(inputs), options=opts)
        system = r.learner.problem
    predict_sec = time.perf_counter() - t0

    t_op = np.asarray(r.t, dtype=float)
    y_op = np.asarray(r.y, dtype=float)                       # (n_seg+1, n_state)
    names = manifest["state_names"]
    out = {
        "x0": list(x0), "inputs": list(inputs),
        "t": finite_list(t_op * manifest["time_factor"]),
        "y": {n: finite_list(y_op[:, i]) for i, n in enumerate(names)},
        "predict_seconds": predict_sec,
        "domain_warnings": warn_domain,
        "reference": None,
        "metrics": {},
    }
    if reference:
        out["reference"], out["metrics"] = _reference(manifest, system, x0, inputs, t_op, y_op)
    return out


def _reference(manifest, system, x0, inputs, t_op, y_op) -> Tuple[Dict, Dict]:
    names = manifest["state_names"]
    t_fine = np.linspace(0.0, float(t_op[-1]), FINE_POINTS)
    try:
        if manifest["source"] == "custom":
            from adalib.utils.reference import solve_reference_ivp
            label = "scipy solve_ivp (BDF) via adalib.utils.reference.solve_reference_ivp, controls held constant"
            u = list(inputs) or None
            y_fine = solve_reference_ivp(system, x0, t_fine, controls=u, method="BDF")
            y_at = solve_reference_ivp(system, x0, t_op, controls=u, method="BDF")
        else:
            label = (f"adalib legacy problem solve_reference ({manifest.get('reference_method')}) — "
                     "the equations the operator was trained on")
            y_fine = system.solve_reference(inputs, x0, t_fine)
            y_at = system.solve_reference(inputs, x0, t_op)
    except Exception as exc:
        return {"label": "reference", "error": f"{type(exc).__name__}: {exc}", "t": None, "y": None}, {}
    y_fine, y_at = np.asarray(y_fine, float), np.asarray(y_at, float)
    metrics = {}
    for i, n in enumerate(names):
        d = y_op[:, i] - y_at[:, i]
        span = float(np.ptp(y_fine[:, i]))
        metrics[n] = {"max_abs": float(np.max(np.abs(d))),
                      "rmse": float(np.sqrt(np.mean(d ** 2))),
                      "max_abs_over_range": float(np.max(np.abs(d)) / span) if span > 0 else None}
    ref = {"label": label, "error": None, "t": finite_list(t_fine * manifest["time_factor"]),
           "y": {n: finite_list(y_fine[:, i]) for i, n in enumerate(names)}}
    return ref, metrics


def _domain_warnings(manifest, x0, inputs) -> List[str]:
    out = []
    for kind, values, names in (("x0", x0, manifest["state_names"]), ("inputs", inputs, manifest["input_names"])):
        for v, (lo, hi), n in zip(values, manifest["domain"][kind], names):
            if v < lo or v > hi:
                out.append(f"{n} = {v:g} is outside the training range [{lo:g}, {hi:g}]; "
                           "the operator is extrapolating.")
    return out


def _warnings(caught) -> List[str]:
    seen, out = set(), []
    for w in caught:
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning)):
            continue
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in seen:
            seen.add(msg)
            out.append(msg)
    return out
