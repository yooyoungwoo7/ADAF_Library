"""
Trained MPC surrogates (import-light: used by the API and the workers).

A surrogate is the checkpoint + data directory that ``adalib.run_mpc`` writes
while training its operator. ``run_mpc(..., reuse_existing_operator=True,
generate_data=False, checkpoint_dir=..., data_dir=...)`` loads it again and
runs only the closed loop, so one training serves any number of MPC runs and
controllers.

Two kinds of job hold one:
  mpc_train  -> ``model.json`` written by the training job (directories under surrogate/)
  mpc (old)  -> a one-shot run from before the split; its request.json fully
                describes the surrogate it trained under mpc/operator/
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas.mpc import MPCSurrogateOptions

MANIFEST = "model.json"

BUILTIN_INFO = {
    "cstr": {"mode": "tracking", "time_unit": "min", "states": ["C_A", "C_B", "T_R", "T_K"], "controls": ["Q_dot"]},
    "triple_tank": {"mode": "tracking", "time_unit": "s", "states": ["h1", "h2", "h3"], "controls": ["Q1", "Q2"]},
    "fedbatch_bioreactor": {"mode": "economic", "time_unit": "min", "states": ["Xs", "Ss", "Ps", "Vs"],
                            "controls": ["inp"]},
}


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _local(ts: Optional[str]) -> Optional[str]:
    """Job timestamps are UTC ISO strings; manifests use local time like training jobs do."""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%dT%H:%M:%S") if ts else None
    except ValueError:
        return ts


def _status(job_dir: Path) -> Optional[dict]:
    return _read(job_dir / "status.json")


def _from_one_shot(job_dir: Path, status: dict) -> Optional[Dict]:
    """Manifest for the surrogate a pre-split ``mpc`` job trained."""
    req = _read(job_dir / "request.json") or {}
    p = req.get("payload") or {}
    if "model_id" in p or "options" not in p:
        return None                                        # a closed-loop-only run: it trained nothing
    work = job_dir / "mpc" / "operator"
    ckpt, data = work / "checkpoints", work / "data"
    if not ckpt.is_dir() or not any(ckpt.iterdir()):
        return None
    o = p["options"]
    m = {
        "model_id": job_dir.name,
        "kind": "mpc_surrogate",
        "origin": "mpc_run",
        "source": p["source"],
        # adalib defaults for anything the old request left out
        "options": {k: o.get(k, v) for k, v in MPCSurrogateOptions.model_construct().model_dump().items()},
        "checkpoint_dir": "mpc/operator/checkpoints",
        "data_dir": "mpc/operator/data",
        "trained_at": _local(status.get("ended_at") or status.get("created_at")),
        "train_seconds": None,
    }
    if p["source"] == "custom":
        spec = p["system"]
        m.update(system_name=spec.get("name") or "custom_system", spec=spec, dt=p["dt"], mode="tracking",
                 time_unit="", builtin=None,
                 state_names=[s["name"].strip() for s in spec["states"]],
                 control_names=[c["name"].strip() for c in spec.get("controls") or []])
    else:
        b = BUILTIN_INFO[p["builtin"]]
        m.update(system_name=p["builtin"], builtin=p["builtin"], spec=None, dt=None, mode=b["mode"],
                 time_unit=b["time_unit"], state_names=b["states"], control_names=b["controls"])
    m["label"] = f"{(status.get('label') or m['system_name']).removesuffix(' (MPC)')} · from MPC run {job_dir.name}"
    return m


def load(job_dir: Path) -> Optional[Dict]:
    """Manifest of the surrogate held by a completed job, or None."""
    st = _status(job_dir)
    if not st or st.get("status") != "completed":
        return None
    if st.get("kind") == "mpc_train":
        return _read(job_dir / MANIFEST)
    if st.get("kind") == "mpc":
        return _from_one_shot(job_dir, st)
    return None


def list_models(runs_dir: Path) -> List[Dict]:
    out = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir():
            m = load(d)
            if m:
                out.append(m)
    return out
