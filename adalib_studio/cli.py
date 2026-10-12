"""``adalib-studio`` command: start the server and open the app in the browser.

    adalib-studio                  # http://127.0.0.1:8000 (next free port if taken)
    adalib-studio --port 8100 --no-browser
    adalib-studio --runs D:\\studio-runs
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from typing import List, Optional

DEFAULT_PORT = 8000
PORT_TRIES = 20


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _studio_at(url: str) -> bool:
    """True when an ADALib Studio server already answers at ``url``."""
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.5) as r:
            return "adalib_installed" in json.load(r)
    except Exception:
        return False


def _open_when_ready(host: str, port: int, url: str):
    for _ in range(120):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    webbrowser.open(url)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="adalib-studio", description="ADALib Studio: browser workspace for adalib (adalib-IFL).")
    ap.add_argument("--port", type=int, default=None, help=f"port to serve on (default {DEFAULT_PORT}, or the next free one)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1: this computer only)")
    ap.add_argument("--runs", default=None, help="directory for job results (default ~/adalib-studio/runs)")
    ap.add_argument("--max-parallel", type=int, default=None, help="jobs computed at the same time (default 1)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)

    # config.py reads these at import time, so set them before importing the app
    if args.runs:
        os.environ["ADALIB_STUDIO_RUNS"] = args.runs
    if args.max_parallel:
        os.environ["ADALIB_STUDIO_MAX_PARALLEL"] = str(args.max_parallel)

    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    if args.port is not None:
        port = args.port
        if not _port_free(args.host, port):
            print(f"Port {port} is already in use. Choose another with --port.", file=sys.stderr)
            return 1
    else:
        port = next((p for p in range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES) if _port_free(args.host, p)), None)
        if port is None:
            print(f"No free port in {DEFAULT_PORT}-{DEFAULT_PORT + PORT_TRIES - 1}. Choose one with --port.", file=sys.stderr)
            return 1
        if port != DEFAULT_PORT and _studio_at(f"http://{browse_host}:{DEFAULT_PORT}"):
            url = f"http://{browse_host}:{DEFAULT_PORT}"
            print(f"ADALib Studio is already running at {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return 0

    import uvicorn

    from .config import FRONTEND_DIST, RUNS_DIR
    from .main import app
    from .services.package_info import adalib_version

    url = f"http://{browse_host}:{port}"
    print(f"ADALib Studio  ->  {url}")
    print(f"  adalib-IFL {adalib_version() or 'NOT INSTALLED'} | results in {RUNS_DIR}")
    if not (FRONTEND_DIST / "index.html").is_file():
        print(f"  warning: no built frontend at {FRONTEND_DIST}; only the API is served", file=sys.stderr)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  warning: bound to {args.host}; other computers on the network can reach this server", file=sys.stderr)
    print("  Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(browse_host, port, url), daemon=True).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
