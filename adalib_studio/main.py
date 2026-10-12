"""ADALib Studio API server: ``uvicorn app.main:app`` from the backend directory, or ``adalib-studio`` when installed with adalib-IFL."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import CORS_ORIGINS, FRONTEND_DIST
from .services.job_manager import JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = JobManager()
    yield
    app.state.jobs.shutdown()


app = FastAPI(title="ADALib Studio API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

# Serve the built frontend (npm run build, or the copy bundled in adalib-IFL) from the same origin when present.
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = (FRONTEND_DIST / path).resolve()
        if path and target.is_file() and FRONTEND_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")
