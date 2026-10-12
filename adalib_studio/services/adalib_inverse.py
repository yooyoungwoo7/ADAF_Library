"""
Parameter estimation via adalib.run_inverse (runs inside worker processes only).

The system is built by the same parser as the Forward solver, which supplies a
TF-native ``rhs_tf``. adalib 0.1.0 requires that for inverse training: the
numpy bridge breaks gradients with respect to the parameters
(docs/adalib_operator_inverse_mpc.md §2.3).

Observations either come from the user's CSV (mapped to states in the UI) or
are generated here from an *independent* scipy Radau solution with Gaussian
noise, as adalib's own examples/inverse/lotka_volterra_inverse.py does to avoid
fitting ADA-generated data with the ADA representation ("inverse crime").
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..schemas.inverse import InverseRunRequest, clean_observations
from .result_serializer import downsample_index, finite_list, write_csv
from .system_builder import build_callable_system

MAX_POINTS = 20_000
MAX_HISTORY = 4_000
SYNTH_LABEL = "scipy solve_ivp (Radau, rtol=1e-11, atol=1e-12) + Gaussian noise — independent of ADA"


def run_inverse_job(req: InverseRunRequest, job_dir: Path) -> Dict:
    import adalib
    from adalib import InverseOptions, InverseParameter, ObservationData, run_inverse

    system, x0, values = build_callable_system(req.system)
    names = system.state_names
    pnames = system.parameter_names
    unknown = {u.name: u for u in req.unknowns}
    t_span = (req.time.t0, req.time.t1)
    notes: List[str] = []

    # ── observations ─────────────────────────────────────────────────
    if req.data_source == "synthetic":
        syn = req.synthetic
        true_vals = dict(zip(pnames, values))
        true_vals.update(syn.true_values)
        observed = syn.observed or names
        t_obs, y_obs = _synthetic(system, x0, t_span, [true_vals[n] for n in pnames], observed, syn)
        source = {"kind": "synthetic", "label": SYNTH_LABEL, "n_points": syn.n_points,
                  "noise_pct": syn.noise_pct, "seed": syn.seed, "true_values": true_vals}
        reference_values = {n: true_vals[n] for n in unknown}
    else:
        t_list, cols, notes = clean_observations(req.upload, req.time)
        observed = [n for n in names if n in cols]
        t_obs = np.asarray(t_list, dtype=float)
        y_obs = np.stack([np.asarray(cols[n], dtype=float) for n in observed], axis=1)
        source = {"kind": "upload", "label": f"uploaded file {req.upload.filename}".strip(), "n_points": len(t_list)}
        reference_values = dict(req.reference_values)
    idx = [names.index(n) for n in observed]
    obs = ObservationData(t=t_obs, y=y_obs, state_indices=idx)

    # ── parameters: ordered like the system, unknowns as InverseParameter ──
    params = {}
    for n, v in zip(pnames, values):
        if n in unknown:
            u = unknown[n]
            params[n] = InverseParameter(initial=u.initial, lower=u.lower, upper=u.upper)
        else:
            params[n] = float(v)

    opts = InverseOptions(**req.options.model_dump(), verbose=True)
    print(f"[studio] adalib {adalib.__version__}: run_inverse(system={system.name!r}, unknown={list(unknown)}, "
          f"fixed={ {n: v for n, v in params.items() if n not in unknown} }, observations={len(t_obs)} x {observed})",
          flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        result = run_inverse(system, x0=x0, t_span=t_span, params=params, data=obs, options=opts)
        wall = time.perf_counter() - t0

    t, y = (np.asarray(a, dtype=float) for a in result.to_arrays())      # (Nt,), (n_state, Nt)
    est = {k: float(v) for k, v in result.estimated_params.items()}

    # fit residuals at the observation times (ADA trajectory interpolated linearly)
    y_fit = np.stack([np.interp(t_obs, t, y[i]) for i in idx], axis=1)
    resid = y_obs - y_fit
    fit = {n: {"rmse": float(np.sqrt(np.mean(resid[:, k] ** 2))),
               "max_abs": float(np.max(np.abs(resid[:, k])))} for k, n in enumerate(observed)}

    # trajectory of the fitted model, integrated independently (verification)
    check = _reference_with(system, x0, t, [est.get(n, params[n]) if n in unknown else params[n] for n in pnames], names)

    table = []
    for n in pnames:
        row = {"name": n, "estimated": n in unknown, "value": est.get(n, params[n] if n not in unknown else None)}
        if n in unknown:
            u = unknown[n]
            row.update(initial=u.initial, lower=u.lower, upper=u.upper)
            ref = reference_values.get(n)
            if ref is not None:
                row.update(reference=ref, abs_error=abs(est[n] - ref),
                           rel_error=abs(est[n] - ref) / abs(ref) if ref != 0 else None)
        table.append(row)

    write_csv(job_dir / "trajectory.csv", {"t": t, **{n: y[i] for i, n in enumerate(names)}})
    write_csv(job_dir / "observations.csv", {"t": t_obs, **{n: y_obs[:, k] for k, n in enumerate(observed)}})
    hist_cols = {"step": np.arange(len(result.loss_history)), "total_loss": result.loss_history,
                 "physics_loss": result.physics_loss_history, "data_loss": result.data_loss_history}
    write_csv(job_dir / "loss_history.csv", hist_cols)
    if result.parameter_history:
        n_ph = min(len(v) for v in result.parameter_history.values())
        write_csv(job_dir / "parameter_history.csv",
                  {"step": np.arange(n_ph), **{k: np.asarray(v)[:n_ph] for k, v in result.parameter_history.items()}})

    di = downsample_index(len(t), MAX_POINTS)
    return {
        "kind": "inverse",
        "adalib_version": adalib.__version__,
        "system": {"name": system.name, "state_names": names, "parameter_names": pnames, "x0": x0},
        "t_span": list(t_span),
        "options": req.options.model_dump(),
        "parameters": table,
        "estimated": est,
        "observations": {"t": finite_list(t_obs), "y": {n: finite_list(y_obs[:, k]) for k, n in enumerate(observed)},
                         "source": source, "notes": notes},
        "t": finite_list(t[di]),
        "y": {n: finite_list(y[i][di]) for i, n in enumerate(names)},
        "fit": fit,
        "residuals": {"t": finite_list(t_obs), "y": {n: finite_list(resid[:, k]) for k, n in enumerate(observed)}},
        "check": check,
        "history": _histories(result),
        "runtime_sec": float(result.runtime_sec),
        "wall_time_sec": wall,
        "warnings": _warnings(caught),
        "files": ["trajectory.csv", "observations.csv", "loss_history.csv", "parameter_history.csv"],
    }


def _synthetic(system, x0, t_span, true_params, observed, syn):
    from adalib.utils.reference import solve_reference_ivp

    names = system.state_names
    t_dense = np.linspace(t_span[0], t_span[1], max(4000, syn.n_points * 4))
    y_dense = solve_reference_ivp(system, x0, t_dense, params=true_params, method="Radau", rtol=1e-11, atol=1e-12)
    # sample like the adalib example: skip t0 (the known initial condition)
    sel = np.linspace(1, t_dense.size - 1, syn.n_points).astype(int)
    t_obs = t_dense[sel]
    idx = [names.index(n) for n in observed]
    clean = y_dense[sel][:, idx]
    rms = np.sqrt(np.mean(y_dense[:, idx] ** 2, axis=0))
    rng = np.random.default_rng(syn.seed)
    noisy = clean + rng.normal(0.0, 1.0, clean.shape) * (syn.noise_pct / 100.0 * rms)[None, :]
    print(f"[studio] synthetic observations: {syn.n_points} points of {observed}, noise {syn.noise_pct:g} % RMS, "
          f"seed {syn.seed} ({SYNTH_LABEL})", flush=True)
    return t_obs, noisy


def _reference_with(system, x0, t, p, names) -> Dict:
    from adalib.utils.reference import solve_reference_ivp

    label = "scipy BDF with the estimated parameters (adalib.utils.reference.solve_reference_ivp)"
    try:
        y_ref = solve_reference_ivp(system, x0, t, params=p, method="BDF")
    except Exception as exc:
        return {"label": label, "error": f"{type(exc).__name__}: {exc}", "y": None}
    di = downsample_index(len(t), MAX_POINTS)
    return {"label": label, "error": None, "y": {n: finite_list(y_ref[di, i]) for i, n in enumerate(names)}}


def _histories(result) -> Dict:
    n = len(result.loss_history)
    di = downsample_index(n, MAX_HISTORY)
    out = {"step": [int(i) for i in di], "n_steps": n,
           "total": finite_list(np.asarray(result.loss_history)[di]),
           "physics": finite_list(np.asarray(result.physics_loss_history)[di]),
           "data": finite_list(np.asarray(result.data_loss_history)[di]), "params": {}}
    for k, v in (result.parameter_history or {}).items():
        v = np.asarray(v, dtype=float)
        pi = downsample_index(len(v), MAX_HISTORY)
        out["params"][k] = {"step": [int(i) for i in pi], "value": finite_list(v[pi])}
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
