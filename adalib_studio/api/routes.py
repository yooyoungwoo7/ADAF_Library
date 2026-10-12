from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

import json

from ..examples.catalog import MPC_BUILTINS, MPC_PRESETS, OPERATOR_BUILTINS, OPERATOR_PRESETS, list_examples
from ..schemas.assist import (AssistForwardRequest, AssistForwardResponse, AssistModuleRequest, AssistModuleResponse,
                              AssistSettingsUpdate, AssistStatus)
from ..schemas.common import JobCreated, SystemSpec, ValidateResponse
from ..schemas.forward import ForwardRunRequest
from ..schemas.inverse import InverseRunRequest
from ..schemas.mpc import MPCLoopRequest, MPCTrainRequest
from ..schemas.operator import OperatorInferRequest, OperatorTrainRequest
from ..services import llm_assist, mpc_models, package_info
from ..services.capabilities import capabilities
from ..services.equation_parser import EquationError
from ..services.log_parser import (forward_progress, inverse_progress, mpc_progress, operator_progress,
                                   parse_forward_log, parse_operator_log)
from ..services.system_builder import parse_spec, validate_spec

router = APIRouter(prefix="/api")


def _jobs(request: Request):
    return request.app.state.jobs


@router.get("/health")
def health(request: Request):
    version = package_info.adalib_version()
    return {"status": "ok", "adalib_installed": version is not None, "adalib_version": version,
            "adalib_version_supported": package_info.version_supported(),
            "dependencies": package_info.dependency_versions(),
            # lets `adalib-studio` tell its own running server from another installation
            "runs_dir": str(_jobs(request).runs_dir)}


@router.get("/capabilities")
def get_capabilities():
    return capabilities()


@router.get("/examples")
def examples(module: str | None = None):
    return list_examples(module)


@router.post("/system/validate", response_model=ValidateResponse)
def validate_system(spec: SystemSpec):
    return validate_spec(spec)


# ── natural-language assistant (LLM) ────────────────────────────────────

@router.get("/assist/status", response_model=AssistStatus)
def assist_status():
    return llm_assist.status()


@router.put("/assist/settings", response_model=AssistStatus)
def assist_settings(upd: AssistSettingsUpdate):
    return llm_assist.update_settings(upd)


@router.post("/assist/forward", response_model=AssistForwardResponse)
def assist_forward(req: AssistForwardRequest):
    try:
        return llm_assist.describe_forward(req)
    except llm_assist.AssistError as exc:
        raise HTTPException(exc.status, detail={"message": str(exc)})


@router.post("/assist/{module}", response_model=AssistModuleResponse)
def assist_module(module: Literal["operator", "inverse", "mpc"], req: AssistModuleRequest):
    try:
        return llm_assist.describe_module(module, req)
    except llm_assist.AssistError as exc:
        raise HTTPException(exc.status, detail={"message": str(exc)})


# ── forward ─────────────────────────────────────────────────────────────

@router.post("/forward/run", response_model=JobCreated, status_code=202)
def forward_run(req: ForwardRunRequest, request: Request):
    if req.system.controls:
        raise HTTPException(422, detail={"message": "The Forward solver has no control inputs; "
                                                    "write their values into the equations or use parameters."})
    _parse_or_422(req.system)
    st = _jobs(request).submit("forward", req.model_dump(), label=req.system.name)
    return JobCreated(job_id=st["id"], status=st["status"])


def _parse_or_422(spec: SystemSpec):
    try:
        parse_spec(spec)
    except EquationError as exc:
        raise HTTPException(422, detail={"message": str(exc), "index": exc.index, "field": exc.field_name})


# ── inverse ─────────────────────────────────────────────────────────────

@router.post("/inverse/run", response_model=JobCreated, status_code=202)
def inverse_run(req: InverseRunRequest, request: Request):
    _parse_or_422(req.system)
    if req.data_source == "upload":
        from ..schemas.inverse import clean_observations
        try:
            clean_observations(req.upload, req.time)
        except ValueError as exc:
            raise HTTPException(422, detail={"message": f"Observations: {exc}"})
    st = _jobs(request).submit("inverse", req.model_dump(), label=f"{req.system.name} (inverse)")
    return JobCreated(job_id=st["id"], status=st["status"])


# ── MPC ─────────────────────────────────────────────────────────────────

@router.get("/mpc/builtins")
def mpc_builtins():
    return {"systems": MPC_BUILTINS, "presets": MPC_PRESETS}


@router.post("/mpc/train", response_model=JobCreated, status_code=202)
def mpc_train(req: MPCTrainRequest, request: Request):
    """Train the surrogate only; the closed loop runs separately on the saved model."""
    if req.source == "custom":
        _parse_or_422(req.system)
    label = req.label or (req.system.name if req.source == "custom" else req.builtin)
    st = _jobs(request).submit("mpc_train", req.model_dump(), label=f"{label} (MPC surrogate)")
    return JobCreated(job_id=st["id"], status=st["status"])


@router.get("/mpc/models")
def mpc_model_list(request: Request):
    return mpc_models.list_models(_jobs(request).runs_dir)


@router.post("/mpc/run", response_model=JobCreated, status_code=202)
def mpc_run(req: MPCLoopRequest, request: Request):
    """Closed loop(s) on a trained surrogate: every selected controller reuses the same model."""
    try:
        _jobs(request).get(req.model_id)
    except ValueError:
        raise HTTPException(404, detail={"message": f"No MPC surrogate {req.model_id!r}."})
    m = mpc_models.load(_jobs(request).runs_dir / req.model_id)
    if m is None:
        raise HTTPException(404, detail={"message": f"No trained MPC surrogate {req.model_id!r}."})
    states = m["state_names"]
    if len(req.x0) != len(states):
        raise HTTPException(422, detail={"message": f"The surrogate expects {len(states)} initial values ({', '.join(states)})."})
    if m["source"] == "custom":
        if not any(n in states for n in req.targets):
            raise HTTPException(422, detail={"message": "Choose at least one state to track."})
        if any(c.optimizer != "SLSQP" for c in req.controllers):
            raise HTTPException(422, detail={"message": "Custom systems use SLSQP (FD or AD) in adalib 0.1.0; "
                                                        "CEM/MPPI are built-in only."})
    ctrl = ", ".join(c.label for c in req.controllers)
    st = _jobs(request).submit("mpc", req.model_dump(), label=f"{m['system_name']} (MPC · {ctrl})")
    return JobCreated(job_id=st["id"], status=st["status"])


# ── operator ────────────────────────────────────────────────────────────

@router.get("/operator/builtins")
def operator_builtins():
    return {"systems": OPERATOR_BUILTINS, "presets": OPERATOR_PRESETS}


@router.post("/operator/train", response_model=JobCreated, status_code=202)
def operator_train(req: OperatorTrainRequest, request: Request):
    if req.source == "custom":
        _parse_or_422(req.system)
    label = req.label or (req.system.name if req.source == "custom" else req.builtin)
    st = _jobs(request).submit("operator_train", req.model_dump(), label=f"{label} (operator)")
    return JobCreated(job_id=st["id"], status=st["status"])


def _manifest(request: Request, model_id: str):
    try:
        st = _jobs(request).get(model_id)
    except ValueError:
        st = None
    if not st or st["kind"] != "operator_train" or st["status"] != "completed":
        return None
    path = _jobs(request).file_path(model_id, "model.json")
    return json.loads(path.read_text(encoding="utf-8")) if path else None


@router.get("/operator/models")
def operator_models(request: Request):
    out = []
    for st in _jobs(request).list(1000):
        if st["kind"] == "operator_train" and st["status"] == "completed":
            m = _manifest(request, st["id"])
            if m:
                out.append(m)
    return out


@router.post("/operator/infer", response_model=JobCreated, status_code=202)
def operator_infer(req: OperatorInferRequest, request: Request):
    m = _manifest(request, req.model_id)
    if m is None:
        raise HTTPException(404, detail={"message": f"No completed operator model {req.model_id!r}."})
    if len(req.x0) != len(m["state_names"]) or len(req.inputs) != len(m["input_names"]):
        raise HTTPException(422, detail={"message": f"Model expects {len(m['state_names'])} initial values and "
                                                    f"{len(m['input_names'])} inputs."})
    st = _jobs(request).submit("operator_infer", req.model_dump(), label=f"{m['label']} (prediction)")
    return JobCreated(job_id=st["id"], status=st["status"])


# ── jobs ────────────────────────────────────────────────────────────────

def _status_or_404(request: Request, job_id: str) -> dict:
    try:
        st = _jobs(request).get(job_id)
    except ValueError:
        st = None
    if st is None:
        raise HTTPException(404, detail="job not found")
    return st


@router.get("/jobs")
def list_jobs(request: Request, limit: int = Query(100, ge=1, le=1000)):
    return _jobs(request).list(limit)


@router.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    st = _status_or_404(request, job_id)
    if st["kind"] == "forward":
        st["progress"] = forward_progress(_jobs(request).full_log(job_id))
    elif st["kind"] == "operator_train":
        st["progress"] = operator_progress(_jobs(request).full_log(job_id))
    elif st["kind"] == "inverse":
        st["progress"] = inverse_progress(_jobs(request).full_log(job_id))
    elif st["kind"] == "mpc":
        p = (_jobs(request).request(job_id) or {}).get("payload") or {}
        n_steps = p.get("n_steps") or (p.get("options") or {}).get("n_steps")
        st["progress"] = mpc_progress(_jobs(request).full_log(job_id), n_steps)
    elif st["kind"] == "mpc_train":
        st["progress"] = mpc_progress(_jobs(request).full_log(job_id), None)
    return st


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, request: Request, offset: int = Query(0, ge=0)):
    _status_or_404(request, job_id)
    text, new_offset = _jobs(request).log(job_id, offset)
    return {"text": text, "offset": new_offset}


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str, request: Request):
    _status_or_404(request, job_id)
    return _jobs(request).cancel(job_id)


@router.delete("/jobs/{job_id}")
def job_delete(job_id: str, request: Request):
    _status_or_404(request, job_id)
    try:
        _jobs(request).delete(job_id)
    except RuntimeError as exc:
        raise HTTPException(409, detail={"message": str(exc)})
    return {"deleted": job_id}


@router.get("/jobs/{job_id}/request")
def job_request(job_id: str, request: Request):
    _status_or_404(request, job_id)
    return _jobs(request).request(job_id)


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, request: Request):
    st = _status_or_404(request, job_id)
    jobs = _jobs(request)
    if st["status"] == "failed":
        raise HTTPException(409, detail={"status": "failed", "error": st.get("error"),
                                         "detail": jobs.error(job_id)})
    if st["status"] != "completed":
        raise HTTPException(409, detail={"status": st["status"], "error": "job has not completed"})
    result = jobs.result(job_id)
    if st["kind"] == "forward":
        # Training history = the per-stage losses adalib printed (no history object exists in 0.1.0)
        result["history"] = parse_forward_log(jobs.full_log(job_id))
    elif st["kind"] in ("operator_train", "mpc_train") and result.get("history") is None:
        # generic path / MPC surrogate: adalib keeps no history object; use its logged epochs
        recs = parse_operator_log(jobs.full_log(job_id))
        result["history"] = {"epoch": [r["epoch"] for r in recs], "phys_loss": [r["loss"] for r in recs],
                             "val_phys_loss": [r["val_loss"] for r in recs], "source": "log"}
    return result


@router.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, request: Request):
    _status_or_404(request, job_id)
    if name not in {"trajectory.csv", "request.json", "result.json", "log.txt", "model.json",
                    "observations.csv", "loss_history.csv", "parameter_history.csv",
                    "closed_loop.csv", "controls.csv"}:
        raise HTTPException(404, detail="file not available")
    path = _jobs(request).file_path(job_id, name)
    if path is None:
        raise HTTPException(404, detail="file not found")
    return FileResponse(path, filename=f"{job_id}_{name}")
