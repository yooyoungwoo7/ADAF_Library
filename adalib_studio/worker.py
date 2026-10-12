"""
Worker entry point: ``python -u -m app.worker <job_dir>``.

One job = one process. adalib keeps process-global state (sys.path entries,
legacy config modules), so jobs never share an interpreter. The parent
(JobManager) redirects stdout/stderr to <job_dir>/log.txt and terminates the
process to cancel. Output: result.json on success, error.json on failure.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def main(job_dir: Path) -> int:
    from .services.result_serializer import write_json

    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    kind = request["kind"]
    print(f"[studio] worker pid={os.getpid()} job={job_dir.name} kind={kind}", flush=True)
    print("[studio] importing adalib (loads TensorFlow; first start can take a while)...", flush=True)
    try:
        if kind == "forward":
            from .schemas.forward import ForwardRunRequest
            from .services.adalib_adapter import run_forward_job

            result = run_forward_job(ForwardRunRequest.model_validate(request["payload"]), job_dir)
        elif kind == "operator_train":
            from .schemas.operator import OperatorTrainRequest
            from .services.adalib_operator import run_operator_train_job

            result = run_operator_train_job(OperatorTrainRequest.model_validate(request["payload"]), job_dir)
        elif kind == "operator_infer":
            from .schemas.operator import OperatorInferRequest
            from .services.adalib_operator import run_operator_infer_job

            result = run_operator_infer_job(OperatorInferRequest.model_validate(request["payload"]), job_dir,
                                            runs_dir=job_dir.parent)
        elif kind == "inverse":
            from .schemas.inverse import InverseRunRequest
            from .services.adalib_inverse import run_inverse_job

            result = run_inverse_job(InverseRunRequest.model_validate(request["payload"]), job_dir)
        elif kind == "mpc_train":
            from .schemas.mpc import MPCTrainRequest
            from .services.adalib_mpc import run_mpc_train_job

            result = run_mpc_train_job(MPCTrainRequest.model_validate(request["payload"]), job_dir)
        elif kind == "mpc" and "model_id" in request["payload"]:
            from .schemas.mpc import MPCLoopRequest
            from .services.adalib_mpc import run_mpc_loop_job

            result = run_mpc_loop_job(MPCLoopRequest.model_validate(request["payload"]), job_dir,
                                      runs_dir=job_dir.parent)
        elif kind == "mpc":  # one-shot train + loop (jobs submitted before the split)
            from .schemas.mpc import MPCRunRequest
            from .services.adalib_mpc import run_mpc_job

            result = run_mpc_job(MPCRunRequest.model_validate(request["payload"]), job_dir)
        else:
            raise ValueError(f"Unknown job kind {kind!r}")
    except BaseException as exc:  # report everything, including KeyboardInterrupt
        traceback.print_exc()
        write_json(job_dir / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc()})
        sys.stdout.flush()
        return 1
    write_json(job_dir / "result.json", result)
    print("[studio] result written", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main(Path(sys.argv[1]).resolve()))
