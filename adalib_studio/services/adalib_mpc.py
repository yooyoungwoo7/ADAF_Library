"""
Closed-loop MPC via adalib.run_mpc (runs inside worker processes only).

The surrogate operator is trained once (``run_mpc_train_job``,
``MPCOptions(run_closed_loop=False)``) and reused by every closed-loop run
(``run_mpc_loop_job``, ``reuse_existing_operator=True``), so several
controllers -- SLSQP with FD or autodiff gradients, CEM, MPPI -- can be
compared on one trained surrogate. ``run_mpc_job`` is the original one-shot
path (train + loop), kept for jobs submitted before the split. ``x`` is
always the plant response to the computed controls. Two paths
(docs/adalib_operator_inverse_mpc.md §3):

  custom  -> CallableODESystem, generic tracking MPC (data-driven LPA surrogate,
             SLSQP, objective  Σ w_Q (y − y_ref)² + w_R u²  with terminal weight).
  builtin -> legacy MPC problems: cstr and triple_tank (tracking),
             fedbatch_bioreactor (economic: maximise final Ps·Vs).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..schemas.common import SystemSpec
from ..schemas.mpc import MPCLoopRequest, MPCRunRequest, MPCTrainRequest
from . import mpc_models
from .result_serializer import finite_list, write_csv, write_json
from .system_builder import build_operator_system

# Built-in MPC problems: what adalib expects for targets/bounds, and display units.
BUILTIN = {
    "cstr": {"mode": "tracking", "time_unit": "min", "tracked": ["T_R"], "default_target": {"T_R": 136.0},
             "default_bounds": {"Q_dot": (-8500.0, 0.0)}},
    "triple_tank": {"mode": "tracking", "time_unit": "s", "tracked": ["h3"],
                    "default_target": {"h3": 150.0}, "default_bounds": {"Q1": (0.0, 100.0), "Q2": (0.0, 100.0)}},
    "fedbatch_bioreactor": {"mode": "economic", "time_unit": "min", "tracked": [], "default_target": {},
                            "default_bounds": {"inp": (0.005, 0.2)}},
}


def _legacy_bounds(name: str, bounds: Dict[str, tuple]):
    """Studio {control: (lo, hi)} -> the key format each legacy loop reads."""
    if not bounds:
        return None
    if name == "cstr":
        lo, hi = bounds.get("Q_dot", BUILTIN["cstr"]["default_bounds"]["Q_dot"])
        return {"Q_lo": lo, "Q_hi": hi}
    if name == "triple_tank":
        out = {}
        for q in ("Q1", "Q2"):
            lo, hi = bounds.get(q, BUILTIN["triple_tank"]["default_bounds"][q])
            out.update({f"{q}_lo": lo, f"{q}_hi": hi})
        return out
    lo, hi = bounds.get("inp", BUILTIN["fedbatch_bioreactor"]["default_bounds"]["inp"])
    return {"inp_min": lo, "inp_max": hi}


def run_mpc_job(req: MPCRunRequest, job_dir: Path) -> Dict:
    import adalib
    from adalib import MPCOptions, run_mpc

    o = req.options
    common = dict(n_steps=o.n_steps, horizon=o.horizon, gradient=o.gradient, optimizer=o.optimizer,
                  n_train=o.n_train, n_val=o.n_val, epochs=o.epochs, batch_size=o.batch_size, lr=o.lr,
                  hidden=o.hidden, n_layers=o.n_layers, seed=o.seed, use_lr_schedule=o.use_lr_schedule,
                  work_dir=str((job_dir / "mpc").resolve()), verbose=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        if req.source == "custom":
            spec = req.system
            system = build_operator_system(spec)
            tracked = [n for n in spec.state_names if n in req.targets]
            controls = spec.control_names
            x0 = [float(s.initial) for s in spec.states]
            opts = MPCOptions(**common, mode="tracking", target={n: req.targets[n] for n in tracked},
                              controlled_variables=tracked, dt=req.dt, plant_solver=o.plant_solver,
                              tracking_weights=[req.tracking_weights.get(n, 1.0) for n in tracked],
                              control_weights=[req.control_weights.get(n, 0.01) for n in controls],
                              lpa_n_panels=o.lpa_n_panels, lpa_max_order=o.lpa_max_order, lpa_nt_seg=o.lpa_nt_seg)
            print(f"[studio] adalib {adalib.__version__}: run_mpc(CallableODESystem {system.name!r}, tracking "
                  f"{opts.target}, dt={req.dt:g}, n_steps={o.n_steps}, horizon={o.horizon or 1})", flush=True)
            result = run_mpc(system, x0, options=opts)
            mode, time_unit, targets = "tracking", "", dict(opts.target)
            bounds = {c.name: (c.lower, c.upper) for c in spec.controls}
        else:
            b = BUILTIN[req.builtin]
            bounds = {**b["default_bounds"], **req.control_bounds}
            targets = {**b["default_target"], **req.targets} if b["mode"] == "tracking" else {}
            opts = MPCOptions(**common, mode=b["mode"], basis=o.basis, target=targets or None,
                              control_bounds=_legacy_bounds(req.builtin, req.control_bounds))
            print(f"[studio] adalib {adalib.__version__}: run_mpc({req.builtin!r}, {b['mode']}, target={targets}, "
                  f"control_bounds={opts.control_bounds}, n_steps={o.n_steps})", flush=True)
            result = run_mpc(req.builtin, list(req.x0), options=opts)
            mode, time_unit, x0 = b["mode"], b["time_unit"], list(req.x0)
        wall = time.perf_counter() - t0

    system = result.system
    names = list(system.state_names)
    controls = list(system.control_names)
    t = np.asarray(result.t, dtype=float)
    x = np.asarray(result.x, dtype=float)
    u = np.asarray(result.u, dtype=float).reshape(len(result.u), -1)
    cost = np.asarray(result.cost, dtype=float)

    tracking = {}
    for n, ref in targets.items():
        if n in names:
            e = x[:, names.index(n)] - ref
            tracking[n] = {"target": ref, "final_error": float(e[-1]), "final_abs_error": float(abs(e[-1])),
                           "mean_abs_error": float(np.mean(np.abs(e[1:]))) if len(e) > 1 else None}
    economic = None
    if mode == "economic" and {"Ps", "Vs"} <= set(names):
        ps, vs = x[:, names.index("Ps")], x[:, names.index("Vs")]
        economic = {"label": "Ps x Vs (product amount)", "final": float(ps[-1] * vs[-1]),
                    "series": finite_list(ps * vs)}

    write_csv(job_dir / "closed_loop.csv", {"t": t, **{n: x[:, i] for i, n in enumerate(names)}})
    write_csv(job_dir / "controls.csv", {"t": t[:-1], **{c: u[:, j] for j, c in enumerate(controls)},
                                         "cost": cost})
    meta = {k: v for k, v in (result.metadata or {}).items()
            if isinstance(v, (int, float, str, bool)) and k not in ("best_checkpoint",)}
    return {
        "kind": "mpc",
        "adalib_version": adalib.__version__,
        "source": req.source,
        "system": {"name": getattr(system, "name", req.builtin), "state_names": names, "control_names": controls,
                   "x0": x0},
        "mode": mode,
        "time_unit": time_unit,
        "t": finite_list(t),
        "x": {n: finite_list(x[:, i]) for i, n in enumerate(names)},
        "u": {c: finite_list(u[:, j]) for j, c in enumerate(controls)},
        "cost": finite_list(cost),
        "targets": targets,
        "control_bounds": {k: list(v) for k, v in bounds.items()},
        "tracking": tracking,
        "economic": economic,
        "options": o.model_dump(),
        "metadata": meta,
        "wall_time_sec": wall,
        "warnings": _warnings(caught),
        "files": ["closed_loop.csv", "controls.csv"],
    }


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


# ── split workflow ──────────────────────────────────────────────────────────

def run_mpc_train_job(req: MPCTrainRequest, job_dir: Path) -> Dict:
    """Train the surrogate only and describe it in ``model.json``."""
    import adalib
    from adalib import MPCOptions, run_mpc

    o = req.options
    work = job_dir / "surrogate"
    common = dict(n_train=o.n_train, n_val=o.n_val, epochs=o.epochs, batch_size=o.batch_size, lr=o.lr,
                  hidden=o.hidden, n_layers=o.n_layers, seed=o.seed, use_lr_schedule=o.use_lr_schedule,
                  work_dir=str(work.resolve()), run_closed_loop=False, verbose=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        if req.source == "custom":
            spec = req.system
            system = build_operator_system(spec)
            # the surrogate depends on the state/control box and dt only; a target is required by the
            # generic path's argument checks but unused when run_closed_loop=False
            first = spec.state_names[0]
            opts = MPCOptions(**common, mode="tracking", dt=req.dt, target={first: float(spec.states[0].initial)},
                              controlled_variables=[first], lpa_n_panels=o.lpa_n_panels,
                              lpa_max_order=o.lpa_max_order, lpa_nt_seg=o.lpa_nt_seg)
            print(f"[studio] adalib {adalib.__version__}: run_mpc(CallableODESystem {system.name!r}, dt={req.dt:g}, "
                  f"run_closed_loop=False) -- surrogate training only", flush=True)
            result = run_mpc(system, [float(s.initial) for s in spec.states], options=opts)
            info = {"system_name": spec.name or "custom_system", "builtin": None, "spec": spec.model_dump(),
                    "dt": req.dt, "mode": "tracking", "time_unit": "",
                    "state_names": spec.state_names, "control_names": spec.control_names}
        else:
            b = BUILTIN[req.builtin]
            bi = mpc_models.BUILTIN_INFO[req.builtin]
            opts = MPCOptions(**common, mode=b["mode"], basis=o.basis, target=dict(b["default_target"]) or None)
            print(f"[studio] adalib {adalib.__version__}: run_mpc({req.builtin!r}, run_closed_loop=False) -- "
                  f"surrogate training only", flush=True)
            result = run_mpc(req.builtin, [0.0] * len(bi["states"]), options=opts)
            info = {"system_name": req.builtin, "builtin": req.builtin, "spec": None, "dt": None, "mode": b["mode"],
                    "time_unit": b["time_unit"], "state_names": bi["states"], "control_names": bi["controls"]}
        train_sec = time.perf_counter() - t0

    op = result.operator_result or {}
    paths = result.paths or {}

    def rel(key: str) -> str:
        return Path(paths[key]).resolve().relative_to(job_dir.resolve()).as_posix()

    manifest = {
        "model_id": job_dir.name,
        "kind": "mpc_surrogate",
        "origin": "mpc_train",
        "source": req.source,
        **info,
        "label": req.label or info["system_name"],
        "options": o.model_dump(),
        "checkpoint_dir": rel("checkpoint_dir"),
        "data_dir": rel("data_dir"),
        "best_val_loss": op.get("best_val_loss"),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_seconds": train_sec,
        "adalib_version": adalib.__version__,
    }
    write_json(job_dir / mpc_models.MANIFEST, manifest)
    print(f"[studio] surrogate saved ({manifest['checkpoint_dir']}); run the closed loop from the MPC module",
          flush=True)
    return {"kind": "mpc_train", "model_id": job_dir.name, "manifest": manifest, "train_seconds": train_sec,
            "warnings": _warnings(caught), "files": [mpc_models.MANIFEST]}


def _run_record(c, result, names: List[str], controls: List[str], targets: Dict[str, float], mode: str,
                wall: float) -> Dict:
    t = np.asarray(result.t, dtype=float)
    x = np.asarray(result.x, dtype=float)
    u = np.asarray(result.u, dtype=float).reshape(len(result.u), -1)
    cost = np.asarray(result.cost, dtype=float)
    tracking = {}
    for n, ref in targets.items():
        if n in names:
            e = x[:, names.index(n)] - ref
            tracking[n] = {"target": ref, "final_error": float(e[-1]), "final_abs_error": float(abs(e[-1])),
                           "mean_abs_error": float(np.mean(np.abs(e[1:]))) if len(e) > 1 else None}
    economic = None
    if mode == "economic" and {"Ps", "Vs"} <= set(names):
        ps, vs = x[:, names.index("Ps")], x[:, names.index("Vs")]
        economic = {"label": "Ps x Vs (product amount)", "final": float(ps[-1] * vs[-1]),
                    "series": finite_list(ps * vs)}
    meta = {k: v for k, v in (result.metadata or {}).items()
            if isinstance(v, (int, float, str, bool)) and k not in ("best_checkpoint",)}
    return {"label": c.label, "gradient": c.gradient, "optimizer": c.optimizer,
            "t": finite_list(t), "x": {n: finite_list(x[:, i]) for i, n in enumerate(names)},
            "u": {k: finite_list(u[:, j]) for j, k in enumerate(controls)}, "cost": finite_list(cost),
            "tracking": tracking, "economic": economic, "metadata": meta, "wall_time_sec": wall,
            "_arrays": (t, x, u, cost)}


def run_mpc_loop_job(req: MPCLoopRequest, job_dir: Path, runs_dir: Path) -> Dict:
    """Closed loop(s) on a trained surrogate: no data generation, no training."""
    import adalib
    from adalib import MPCOptions, run_mpc

    model_dir = runs_dir / req.model_id
    m = mpc_models.load(model_dir)
    if m is None:
        raise FileNotFoundError(f"No trained MPC surrogate {req.model_id!r}")
    o = m["options"]
    names, controls, mode = list(m["state_names"]), list(m["control_names"]), m["mode"]
    surrogate = dict(train_operator=False, reuse_existing_operator=True, generate_data=False,
                     checkpoint_dir=str((model_dir / m["checkpoint_dir"]).resolve()),
                     data_dir=str((model_dir / m["data_dir"]).resolve()),
                     hidden=o["hidden"], n_layers=o["n_layers"], lr=o["lr"], seed=o["seed"])
    if m["source"] == "custom":
        spec = SystemSpec.model_validate(m["spec"])
        tracked = [n for n in names if n in req.targets]
        targets = {n: req.targets[n] for n in tracked}
        bounds = {c.name.strip(): (c.lower, c.upper) for c in spec.controls}
    else:
        b = BUILTIN[m["builtin"]]
        bounds = {**b["default_bounds"], **req.control_bounds}
        targets = {**b["default_target"], **req.targets} if mode == "tracking" else {}
    print(f"[studio] adalib {adalib.__version__}: closed loop on surrogate {req.model_id} ({m['label']}); "
          f"no training. Controllers: {', '.join(c.label for c in req.controllers)}", flush=True)

    runs, caught_all = [], []
    t_start = time.perf_counter()
    for i, c in enumerate(req.controllers):
        print(f"[studio] controller {i + 1}/{len(req.controllers)}: {c.label}", flush=True)
        common = dict(**surrogate, n_steps=req.n_steps, horizon=req.horizon, gradient=c.gradient,
                      optimizer=c.optimizer, work_dir=str((job_dir / f"loop_{i + 1}_{c.label}").resolve()),
                      verbose=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            if m["source"] == "custom":
                opts = MPCOptions(**common, mode="tracking", target=dict(targets), controlled_variables=tracked,
                                  dt=m["dt"], plant_solver=req.plant_solver,
                                  tracking_weights=[req.tracking_weights.get(n, 1.0) for n in tracked],
                                  control_weights=[req.control_weights.get(n, 0.01) for n in controls],
                                  lpa_n_panels=o["lpa_n_panels"], lpa_max_order=o["lpa_max_order"],
                                  lpa_nt_seg=o["lpa_nt_seg"])
                result = run_mpc(build_operator_system(spec), list(req.x0), options=opts)
            else:
                opts = MPCOptions(**common, mode=mode, basis=o.get("basis", "lpa"), target=targets or None,
                                  control_bounds=_legacy_bounds(m["builtin"], req.control_bounds))
                result = run_mpc(m["builtin"], list(req.x0), options=opts)
            wall = time.perf_counter() - t0
        caught_all.extend(caught)
        runs.append(_run_record(c, result, names, controls, targets, mode, wall))

    # one CSV per kind; columns are prefixed by the controller when several ran
    multi = len(runs) > 1

    def col(r: Dict, n: str) -> str:
        return f"{r['label']}:{n}" if multi else n

    t = runs[0]["_arrays"][0]
    cl, ct = {"t": t}, {"t": t[:-1]}
    for r in runs:
        _, x, u, cost = r.pop("_arrays")
        cl.update({col(r, n): x[:, i] for i, n in enumerate(names)})
        ct.update({col(r, k): u[:, j] for j, k in enumerate(controls)})
        ct[col(r, "cost")] = cost
    write_csv(job_dir / "closed_loop.csv", cl)
    write_csv(job_dir / "controls.csv", ct)

    first = runs[0]
    return {
        "kind": "mpc",
        "adalib_version": adalib.__version__,
        "source": m["source"],
        "model_id": req.model_id,
        "surrogate": {k: m.get(k) for k in ("label", "origin", "options", "trained_at", "train_seconds",
                                            "best_val_loss", "dt")},
        "system": {"name": m["system_name"], "state_names": names, "control_names": controls, "x0": list(req.x0)},
        "mode": mode,
        "time_unit": m["time_unit"],
        "targets": targets,
        "control_bounds": {k: list(v) for k, v in bounds.items()},
        "runs": runs,
        # first controller at the top level (the layout of one-shot results)
        **{k: first[k] for k in ("t", "x", "u", "cost", "tracking", "economic", "metadata")},
        "options": {**o, "n_steps": req.n_steps, "horizon": req.horizon, "gradient": first["gradient"],
                    "optimizer": first["optimizer"], "plant_solver": req.plant_solver},
        "wall_time_sec": time.perf_counter() - t_start,
        "warnings": _warnings(caught_all),
        "files": ["closed_loop.csv", "controls.csv"],
    }
