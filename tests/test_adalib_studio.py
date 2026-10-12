"""ADAlib Studio is bundled: the GUI server imports without TensorFlow and serves the built frontend."""
import sys

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


def test_studio_server_serves_gui_without_importing_adalib(tmp_path, monkeypatch):
    monkeypatch.setenv("ADALIB_STUDIO_RUNS", str(tmp_path))
    from adalib_studio.config import FRONTEND_DIST
    from adalib_studio.main import app
    from adalib_studio.services.job_manager import WORKER_MODULE

    assert WORKER_MODULE == "adalib_studio.worker"
    assert (FRONTEND_DIST / "index.html").is_file()
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["adalib_installed"] and health["adalib_version_supported"]
        assert "<!doctype html>" in client.get("/").text.lower()
        ok = client.post("/api/system/validate", json={
            "states": [{"name": "x", "initial": 1.0}], "parameters": [{"name": "k", "value": 2.0}],
            "equations": ["-k*x"]}).json()
        assert ok["ok"], ok
    assert "tensorflow" not in sys.modules     # jobs import adalib/TensorFlow in worker processes only
