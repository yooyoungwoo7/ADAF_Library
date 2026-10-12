"""Process helpers for re-attaching to workers that outlived a server restart."""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional


def process_start_time(pid: int) -> Optional[float]:
    """Creation time (epoch seconds) of a live process, or None if it is gone."""
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
                return None
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                return None
            ticks = (c.dwHighDateTime << 32) | c.dwLowDateTime          # 100 ns since 1601-01-01
            return ticks / 1e7 - 11644473600.0
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    try:
        return os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        return time.time()          # alive, start time unknown


def is_same_process(pid: Optional[int], started_iso: Optional[str], tolerance: float = 30.0) -> bool:
    """True if ``pid`` is alive and was created around ``started_iso`` (guards against PID reuse)."""
    if not pid or not started_iso:
        return False
    t = process_start_time(int(pid))
    if t is None:
        return False
    if os.name != "nt":
        return True
    started = datetime.fromisoformat(started_iso).astimezone(timezone.utc).timestamp()
    return abs(t - started) <= tolerance


class AttachedProcess:
    """Minimal Popen-like handle for a worker started by an earlier server process."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        if self.returncode is None and process_start_time(self.pid) is None:
            self.returncode = -1        # exit code of a non-child process is not available
        return self.returncode

    def wait(self) -> int:
        while self.poll() is None:
            time.sleep(1.0)
        return self.returncode


def kill_tree(proc) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # TerminateProcess on the worker and any children it spawned
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif isinstance(proc, AttachedProcess):
        import signal
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
