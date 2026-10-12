"""
The only module that calls adalib. It runs inside a worker process
(app/worker.py), never in the API server process.

Every function takes a validated request model plus the job directory and
returns a JSON-serialisable result dict; large arrays are also written to
files in the job directory.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..schemas.forward import ForwardRunRequest
from .result_serializer import downsample_index, finite_list, write_csv
from .system_builder import build_callable_system

MAX_JSON_POINTS = 20_000
REFERENCE_LABEL = "scipy solve_ivp (BDF, rtol=1e-8, atol=1e-10) via adalib.utils.reference.solve_reference_ivp"


def run_forward_job(req: ForwardRunRequest, job_dir: Path) -> Dict:
    import adalib
    from adalib import ForwardOptions, run_forward

    system, x0, params = build_callable_system(req.system)
    names = system.state_names
    opts = ForwardOptions(**req.options.model_dump(), verbose=True)
    t_span = (req.time.t0, req.time.t1)

    print(f"[studio] adalib {adalib.__version__}: run_forward(system={system.name!r}, "
          f"x0={x0}, t_span={t_span}, params={dict(zip(system.parameter_names, params))})", flush=True)
    print(f"[studio] options: {req.options.model_dump()}", flush=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_start = time.perf_counter()
        result = run_forward(system, x0=x0, t_span=t_span, params=params or None, options=opts)
        wall = time.perf_counter() - t_start
    warning_msgs = _warning_messages(caught)

    t, y = result.to_arrays()               # (Nt,), (n_state, Nt); adalib returns float32
    if y.shape != (len(names), t.shape[0]):
        raise RuntimeError(f"Unexpected result shape from adalib: t{t.shape}, y{y.shape}")

    reference = None
    if req.reference:
        reference = _reference(system, x0, t, params, names)

    columns = {"t": t}
    columns.update({f"{n}": y[i] for i, n in enumerate(names)})
    if reference and reference.get("y") is not None:
        columns.update({f"{n}_reference": np.asarray(reference["y"][n]) for n in names})
    write_csv(job_dir / "trajectory.csv", columns)

    metrics = {}
    if reference and reference.get("y") is not None:
        metrics = error_metrics({n: y[i] for i, n in enumerate(names)}, reference["y"])

    idx = downsample_index(t.shape[0], MAX_JSON_POINTS)
    if reference and reference.get("y") is not None:
        reference["y"] = {n: finite_list(np.asarray(v)[idx]) for n, v in reference["y"].items()}

    meta = {k: v for k, v in result.solution.meta.items()}
    return {
        "kind": "forward",
        "adalib_version": adalib.__version__,
        "system": {"name": system.name, "state_names": names,
                   "parameter_names": system.parameter_names,
                   "parameters": dict(zip(system.parameter_names, params)), "x0": x0},
        "t_span": list(t_span),
        "options": req.options.model_dump(),
        "n_points": int(t.shape[0]),
        "downsampled": bool(len(idx) < t.shape[0]),
        "t": finite_list(t[idx]),
        "y": {n: finite_list(y[i][idx]) for i, n in enumerate(names)},
        "dtype_returned": str(y.dtype),
        "solver_meta": meta,
        "wall_time_sec": wall,
        "reference": reference,
        "metrics": metrics,
        "warnings": warning_msgs,
        "files": ["trajectory.csv"],
    }


def _reference(system, x0, t, params, names) -> Dict:
    from adalib.utils.reference import solve_reference_ivp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            y_ref = solve_reference_ivp(system, x0, np.asarray(t, dtype=float),
                                        params=params or None, method="BDF")
        except Exception as exc:  # reference is optional; report, don't fail the job
            return {"label": REFERENCE_LABEL, "y": None, "error": f"{type(exc).__name__}: {exc}"}
    y_ref = np.asarray(y_ref, dtype=float).T          # (n_state, Nt)
    if y_ref.shape[1] != len(t):
        return {"label": REFERENCE_LABEL, "y": None,
                "error": "Reference integration stopped early: " + "; ".join(_warning_messages(caught))}
    print(f"[studio] reference computed: {REFERENCE_LABEL}", flush=True)
    return {"label": REFERENCE_LABEL, "y": {n: y_ref[i] for i, n in enumerate(names)},
            "error": None, "warnings": _warning_messages(caught)}


def error_metrics(y: Dict[str, np.ndarray], y_ref: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Per-state relative L2 and max-abs difference vs the reference, on the full grid."""
    out = {}
    for n, a in y.items():
        a = np.asarray(a, dtype=float)
        b = np.asarray(y_ref[n], dtype=float)
        denom = np.linalg.norm(b)
        out[n] = {"rel_l2": float(np.linalg.norm(a - b) / denom) if denom > 0 else None,
                  "max_abs": float(np.max(np.abs(a - b)))}
    return out


_NOISE = (DeprecationWarning, PendingDeprecationWarning)


def _warning_messages(caught) -> List[str]:
    """Warnings worth showing the user; deprecation chatter from TF/Python internals is dropped."""
    seen, out = set(), []
    for w in caught:
        if issubclass(w.category, _NOISE):
            continue
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in seen:
            seen.add(msg)
            out.append(msg)
    return out
