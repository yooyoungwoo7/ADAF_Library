"""
Progress extraction from adalib's verbose stdout.

adalib 0.1.0 has no progress callback. The vendored forward solvers print
one line per optimizer stage:

  ADA-F (pinn_lib/ADAF_seq/core/solver_adaf.py), per segment k of n_seg:
    [seg 3/49] Adam done in 0.264s | loss=5.121e+02 | parts=['3.120e+02', ...]
    [seg 3/49] L-BFGS done in 0.252s | loss=2.865e-02 | parts=[...] | CONVERGENCE: ...
  LPA (pinn_lib/ADAF_seq/core/solver_lpa.py), per Adam epoch, then once:
    [epoch 2/5] Adam done in 1.020s | loss=... | parts=[...]
    [full] L-BFGS done in 3.100s | loss=... | parts=[...] | CONVERGENCE: ...

Only values that actually appear in the log are reported.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional

STAGE_RE = re.compile(
    r"\[(?:(?P<unit>seg|epoch) (?P<k>\d+)/(?P<n>\d+)|full)\] (?P<stage>Adam|L-BFGS) done in "
    r"(?P<sec>[\d.]+)s \| loss=(?P<loss>[^\s|]+) \| parts=(?P<parts>\[[^\]]*\])(?: \| (?P<msg>.*))?"
)


def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def parse_stage_line(line: str) -> Optional[Dict]:
    m = STAGE_RE.search(line)
    if not m:
        return None
    try:
        parts = [float(v) for v in ast.literal_eval(m.group("parts"))]
    except (ValueError, SyntaxError):
        parts = []
    unit = m.group("unit") or "full"
    if unit == "seg":
        # "[seg k/last]" — zero-based index, last = n_seg - 1
        index, total = int(m.group("k")), int(m.group("n")) + 1
    elif unit == "epoch":
        # "[epoch i/epochs]" — one-based
        index, total = int(m.group("k")) - 1, int(m.group("n"))
    else:
        index, total = None, None
    return {
        "unit": {"seg": "segment", "epoch": "epoch", "full": "full"}[unit],
        "index": index,
        "total": total,
        "stage": m.group("stage"),
        "seconds": float(m.group("sec")),
        "loss": _float(m.group("loss")),
        "parts": parts,
        "message": (m.group("msg") or "").strip() or None,
    }


def parse_forward_log(text: str) -> List[Dict]:
    return [rec for rec in (parse_stage_line(l) for l in text.splitlines()) if rec]


def forward_progress(text: str) -> Optional[Dict]:
    """The most recent stage line, or None if training has not reported yet."""
    for line in reversed(text.splitlines()):
        rec = parse_stage_line(line)
        if rec:
            return rec
    return None


# ── operator (adalib.run_operator) ──────────────────────────────────────
# built-in (legacy learner.fit, printed at epoch 1, every 10th and the last):
#   [   1/20] phys=1.821497e+03 | val=2.300639e+03 lr=3.00e-03 | 1.91s
# generic physics-only path (mpc/_generic_mpc.py, every max(1, epochs//10)):
#   [LPA op-physics    1/200]  loss=2.6071e-02  val=2.5515e-02
OP_BUILTIN_RE = re.compile(
    r"^\[\s*(?P<k>\d+)/(?P<n>\d+)\] phys=(?P<loss>\S+)(?: \| val=(?P<val>\S+))?(?: lr=(?P<lr>\S+))?")
OP_GENERIC_RE = re.compile(r"\[LPA op(?:-physics)?\s+(?P<k>\d+)/(?P<n>\d+)\]\s+loss=(?P<loss>\S+)\s+val=(?P<val>\S+)")
OP_STAGE_RE = re.compile(r"^\[run_operator(?:/generic)?\] (?P<msg>(Generating|Slicing|Sampling|Training|Rollout|Loaded|Reused).*)")


def parse_operator_line(line: str) -> Optional[Dict]:
    m = OP_BUILTIN_RE.search(line.strip()) or OP_GENERIC_RE.search(line)
    if not m:
        return None
    g = m.groupdict()
    return {"epoch": int(g["k"]), "epochs": int(g["n"]), "loss": _float(g["loss"]),
            "val_loss": _float(g["val"]) if g.get("val") else None,
            "lr": _float(g["lr"]) if g.get("lr") else None}


def parse_operator_log(text: str) -> List[Dict]:
    return [rec for rec in (parse_operator_line(l) for l in text.splitlines()) if rec]


def operator_progress(text: str) -> Optional[Dict]:
    lines = text.splitlines()
    for line in reversed(lines):
        rec = parse_operator_line(line)
        if rec:
            return {"unit": "epoch", "index": rec["epoch"] - 1, "total": rec["epochs"], "stage": "training",
                    "loss": rec["loss"], "val_loss": rec["val_loss"]}
        m = OP_STAGE_RE.search(line)
        if m:
            return {"unit": "stage", "index": None, "total": None, "stage": m.group("msg")[:80], "loss": None}
    return None


# ── inverse (adalib.run_inverse, inverse/solver.py) ─────────────────────
#   [seg   0/3] loss=7.605e+00  phys=4.331e-01  data=7.172e-01  |  alpha=1.4846  gamma=1.1858
#   multi-pass runs prefix "pass i/n " or "warm i/n "
INV_RE = re.compile(
    r"\[(?:(?P<pass>pass|warm) (?P<pi>\d+)/(?P<pn>\d+)\s+)?seg\s+(?P<k>\d+)/(?P<last>\d+)\] loss=(?P<loss>\S+)"
    r"\s+phys=(?P<phys>\S+)\s+data=(?P<data>\S+)\s+\|\s+(?P<params>.*)$")


def parse_inverse_line(line: str) -> Optional[Dict]:
    m = INV_RE.search(line)
    if not m:
        return None
    params = {}
    for tok in m.group("params").split():
        name, _, val = tok.partition("=")
        v = _float(val)
        if name and v is not None:
            params[name] = v
    return {"segment": int(m.group("k")), "n_segments": int(m.group("last")) + 1,
            "pass": m.group("pass"), "loss": _float(m.group("loss")), "physics": _float(m.group("phys")),
            "data": _float(m.group("data")), "params": params}


def inverse_progress(text: str) -> Optional[Dict]:
    for line in reversed(text.splitlines()):
        rec = parse_inverse_line(line)
        if rec:
            stage = ", ".join(f"{k}={v:.4g}" for k, v in rec["params"].items())
            return {"unit": "segment", "index": rec["segment"], "total": rec["n_segments"],
                    "stage": stage or "training", "loss": rec["loss"]}
    return None


# ── MPC (adalib.run_mpc) ────────────────────────────────────────────────
# surrogate training uses the operator formats above; then one line per step:
#   [step   1]  Q*=   -6336 kJ/h  T_R=133.44°C  J=0.000e+00
#   [step   1]  inp=0.0050 L/min  Ps=0.008  Vs=1.010  Ps×Vs=0.008 g
MPC_CTRL_RE = re.compile(r"\[studio\] controller (?P<k>\d+)/(?P<n>\d+): (?P<label>\S+)")
MPC_STEP_RE = re.compile(r"^\s*\[step\s+(?P<k>\d+)\]\s+(?P<body>.*)$")


def mpc_progress(text: str, n_steps: Optional[int]) -> Optional[Dict]:
    lines = text.splitlines()
    ctrl = ""
    for line in reversed(lines):            # "[studio] controller 2/3: AD" (closed loop on a saved surrogate)
        m = MPC_CTRL_RE.search(line)
        if m:
            ctrl = f"{m.group('label')} ({m.group('k')}/{m.group('n')}) · "
            break
    for line in reversed(lines):            # closed-loop steps take precedence
        m = MPC_STEP_RE.search(line)
        if m:
            body = " ".join(m.group("body").split())
            return {"unit": "step", "index": int(m.group("k")) - 1, "total": n_steps,
                    "stage": ctrl + "closed loop · " + body[:90], "loss": None}
        if ctrl and MPC_CTRL_RE.search(line):
            return {"unit": "stage", "index": None, "total": None, "stage": ctrl + "loading surrogate", "loss": None}
    for line in reversed(lines):            # still training the surrogate
        rec = parse_operator_line(line)
        if rec:
            return {"unit": "epoch", "index": rec["epoch"] - 1, "total": rec["epochs"],
                    "stage": "training surrogate", "loss": rec["loss"], "val_loss": rec["val_loss"]}
        if "[run_mpc" in line:
            return {"unit": "stage", "index": None, "total": None, "stage": line.strip()[:80], "loss": None}
    return None
