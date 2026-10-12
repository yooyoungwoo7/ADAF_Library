"""
Process-per-job execution.

Each job lives in RUNS_DIR/<job_id>/:
    request.json   kind + validated payload
    status.json    lifecycle (pending -> running -> completed | failed | cancelled)
    log.txt        worker stdout+stderr (adalib's verbose output)
    result.json    written by the worker on success
    error.json     written by the worker on failure

The directory is the source of truth, so a later persistent queue only has
to replace the in-memory scheduler below.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..config import BACKEND_DIR, MAX_PARALLEL_JOBS, RUNS_DIR, WORKER_PYTHON
from .proc_util import AttachedProcess, is_same_process, kill_tree
from .result_serializer import write_json

TERMINAL = {"completed", "failed", "cancelled"}

# "app.worker" in the source tree, "adalib_studio.worker" when installed with adalib-IFL.
WORKER_MODULE = __package__.rsplit(".", 1)[0] + ".worker"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, runs_dir: Path = RUNS_DIR, max_parallel: int = MAX_PARALLEL_JOBS,
                 python: str = WORKER_PYTHON):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max(1, max_parallel)
        self.python = python
        self._lock = threading.RLock()
        self._procs: Dict[str, subprocess.Popen] = {}
        self._queue: List[str] = []
        self._recover()
        self._schedule()

    # ── public API ────────────────────────────────────────────────────
    def submit(self, kind: str, payload: dict, label: str = "") -> dict:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.runs_dir / job_id
        d.mkdir(parents=True)
        write_json(d / "request.json", {"kind": kind, "payload": payload})
        status = {"id": job_id, "kind": kind, "label": label, "status": "pending",
                  "created_at": _now(), "started_at": None, "ended_at": None,
                  "returncode": None, "pid": None, "error": None}
        write_json(d / "status.json", status)
        with self._lock:
            self._queue.append(job_id)
        self._schedule()
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[dict]:
        p = self._dir(job_id) / "status.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self, limit: int = 100) -> List[dict]:
        out = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            if (d / "status.json").exists():
                out.append(json.loads((d / "status.json").read_text(encoding="utf-8")))
            if len(out) >= limit:
                break
        return out

    def request(self, job_id: str) -> Optional[dict]:
        p = self._dir(job_id) / "request.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def result(self, job_id: str) -> Optional[dict]:
        p = self._dir(job_id) / "result.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def error(self, job_id: str) -> Optional[dict]:
        p = self._dir(job_id) / "error.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def log(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        p = self._dir(job_id) / "log.txt"
        if not p.exists():
            return "", offset
        with open(p, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
        return data.decode("utf-8", errors="replace"), offset + len(data)

    def full_log(self, job_id: str) -> str:
        return self.log(job_id, 0)[0]

    def file_path(self, job_id: str, name: str) -> Optional[Path]:
        d = self._dir(job_id).resolve()
        p = (d / name).resolve()
        if p.parent != d or not p.is_file():
            return None
        return p

    def cancel(self, job_id: str) -> Optional[dict]:
        with self._lock:
            st = self.get(job_id)
            if st is None or st["status"] in TERMINAL:
                return st
            if job_id in self._queue:
                self._queue.remove(job_id)
                self._update(job_id, status="cancelled", ended_at=_now(), error="Cancelled before start")
                return self.get(job_id)
            proc = self._procs.get(job_id)
            self._update(job_id, cancel_requested=True)
        if proc is not None:
            kill_tree(proc)
        return self.get(job_id)

    def shutdown(self, kill_running: bool = False):
        """Stop scheduling. Running workers are left alone by default: they
        write their own results, and the next server start re-attaches them
        (a restart or --reload must not cost a long training run)."""
        with self._lock:
            procs = list(self._procs.values())
            self._queue.clear()          # still "pending" on disk -> re-queued on next start
        if kill_running:
            for p in procs:
                kill_tree(p)

    # ── scheduling ────────────────────────────────────────────────────
    def _schedule(self):
        with self._lock:
            while self._queue and len(self._procs) < self.max_parallel:
                job_id = self._queue.pop(0)
                self._start(job_id)

    def _start(self, job_id: str):
        d = self._dir(job_id)
        env = dict(os.environ)
        env.update({"PYTHONPATH": str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", ""),
                    "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                    "TF_CPP_MIN_LOG_LEVEL": "2", "MPLBACKEND": "Agg",
                    # adalib's legacy trainer draws a tqdm bar on stderr; it would flood log.txt
                    "TQDM_DISABLE": "1"})
        log = open(d / "log.txt", "ab")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            # cwd = job dir, so anything adalib writes relative to cwd stays with the job
            proc = subprocess.Popen([self.python, "-u", "-m", WORKER_MODULE, str(d)], cwd=str(d),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=env, **kwargs)
        except OSError as exc:
            log.close()
            self._update(job_id, status="failed", ended_at=_now(), error=f"Could not start worker: {exc}")
            return
        self._procs[job_id] = proc
        self._update(job_id, status="running", started_at=_now(), pid=proc.pid)
        threading.Thread(target=self._wait, args=(job_id, proc, log), daemon=True).start()

    def _wait(self, job_id: str, proc, log):
        rc = proc.wait()
        if log is not None:
            log.close()
        with self._lock:
            self._procs.pop(job_id, None)
            st = self.get(job_id) or {}
            d = self._dir(job_id)
            if st.get("cancel_requested"):
                self._update(job_id, status="cancelled", ended_at=_now(), returncode=rc,
                             error="Cancelled by user (worker process terminated)")
            elif rc in (0, -1) and (d / "result.json").exists():   # -1: re-attached, exit code unknown
                self._update(job_id, status="completed", ended_at=_now(), returncode=rc)
            else:
                err = self.error(job_id)
                msg = f"{err['type']}: {err['message']}" if err else f"Worker exited with code {rc}"
                self._update(job_id, status="failed", ended_at=_now(), returncode=rc, error=msg)
        self._schedule()

    # ── helpers ───────────────────────────────────────────────────────
    def _dir(self, job_id: str) -> Path:
        if not job_id or any(c in job_id for c in "/\\.:"):
            raise ValueError("invalid job id")
        return self.runs_dir / job_id

    def _update(self, job_id: str, **fields):
        st = self.get(job_id) or {}
        st.update(fields)
        write_json(self._dir(job_id) / "status.json", st)

    def _recover(self):
        """Pick up jobs a previous server process left behind.

        Workers write their log/result/error files themselves, so a worker that
        is still alive (the server was killed or restarted) is re-attached and
        finishes normally; pending jobs are queued again."""
        pending = []
        for d in sorted(self.runs_dir.iterdir()):
            p = d / "status.json"
            if not p.exists():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            status = st.get("status")
            if status in TERMINAL:
                continue
            if status == "pending":
                pending.append(st["id"])
                continue
            if is_same_process(st.get("pid"), st.get("started_at")):
                proc = AttachedProcess(int(st["pid"]))
                self._procs[st["id"]] = proc
                threading.Thread(target=self._wait, args=(st["id"], proc, None), daemon=True).start()
                continue
            if (d / "result.json").exists():
                st.update(status="completed")
            else:
                st.update(status="failed", error="Server restarted and the worker process was no longer running")
            st["ended_at"] = st.get("ended_at") or _now()
            write_json(p, st)
        self._queue.extend(pending)
