"""Runtime configuration (environment-overridable)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# Directory that holds the package: backend/ in the source tree, site-packages when pip-installed.
# Worker processes get it on PYTHONPATH.
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

# Running from this repo's checkout (backend/app) rather than the copy bundled in adalib-IFL.
SOURCE_TREE = (PROJECT_DIR / "frontend" / "package.json").is_file()

# Built frontend: frontend/dist in the source tree, static/ in the bundled copy.
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist" if SOURCE_TREE else PACKAGE_DIR / "static"

# One directory per job: request.json, status.json, log.txt, result.json, exports.
_DEFAULT_RUNS = PROJECT_DIR / "runs" if SOURCE_TREE else Path.home() / "adalib-studio" / "runs"
RUNS_DIR = Path(os.environ.get("ADALIB_STUDIO_RUNS", _DEFAULT_RUNS)).resolve()

# Interpreter used for worker processes. Must have adalib-IFL installed.
WORKER_PYTHON = os.environ.get("ADALIB_STUDIO_PYTHON", sys.executable)

# adalib keeps process-global state, so every job gets its own process; this
# caps how many of those processes train at the same time (CPU-bound).
MAX_PARALLEL_JOBS = int(os.environ.get("ADALIB_STUDIO_MAX_PARALLEL", "1"))

CORS_ORIGINS = os.environ.get(
    "ADALIB_STUDIO_CORS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

ADALIB_DISTRIBUTION = "adalib-IFL"
SUPPORTED_ADALIB_VERSIONS = ("0.1.0", "0.1.1")
