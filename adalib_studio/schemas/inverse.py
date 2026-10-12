from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .common import FiniteMixin, SystemSpec, TimeSpan


class InverseOptionsModel(FiniteMixin):
    """Subset of ``adalib.InverseOptions`` (0.1.0); defaults equal the package's.
    Only the ADA-F basis exists for inverse training in 0.1.0."""

    lambda_physics: float = Field(1.0, ge=0)
    lambda_data: float = Field(10.0, ge=0)
    n_seg: int = Field(20, ge=1, le=2000)
    N_p: int = Field(5, ge=1, le=500)
    N_m: int = Field(100, ge=1, le=2000)
    Nt_total: int = Field(1000, ge=2, le=200_000)
    gamma: float = Field(0.8, gt=0)
    L: float = 1.0
    epochs: int = Field(5, ge=0, le=10_000)
    adam_inner: int = Field(200, ge=1, le=100_000)
    adam_lr: float = Field(1e-3, gt=0, le=1.0)
    use_lbfgs: bool = True
    n_passes: int = Field(1, ge=1, le=100)
    warm_seg_passes: int = Field(1, ge=1, le=100)
    n_warm_segs: int = Field(3, ge=0, le=2000)
    training_strategy: Literal["joint", "alternating"] = "joint"
    data_prefit_steps: int = Field(0, ge=0, le=100_000)
    normalize_data_loss: bool = True
    normalize_physics_loss: bool = False
    param_log_every: int = Field(1, ge=1)
    dtype: Literal["float32", "float64"] = "float64"


class UnknownParameter(FiniteMixin):
    name: str
    initial: float
    lower: Optional[float] = None
    upper: Optional[float] = None

    @model_validator(mode="after")
    def _bounds(self):
        if self.lower is not None and self.upper is not None and not self.upper > self.lower:
            raise ValueError(f"{self.name}: upper bound must exceed lower bound")
        if self.lower is not None and self.initial <= self.lower:
            raise ValueError(f"{self.name}: initial guess must be above the lower bound")
        if self.upper is not None and self.initial >= self.upper:
            raise ValueError(f"{self.name}: initial guess must be below the upper bound")
        return self


class UploadedObservations(BaseModel):
    """Columns already mapped to state names; ``None`` marks a missing value."""
    t: List[Optional[float]] = Field(..., min_length=2, max_length=200_000)
    columns: Dict[str, List[Optional[float]]] = Field(..., min_length=1)
    filename: str = ""


class SyntheticObservations(FiniteMixin):
    """Observations sampled from an independent scipy Radau solution (as in
    adalib's examples/inverse/lotka_volterra_inverse.py), not from ADA."""
    n_points: int = Field(200, ge=2, le=100_000)
    noise_pct: float = Field(1.0, ge=0, le=100, description="Gaussian noise, % of each state's RMS")
    seed: int = 42
    observed: List[str] = Field(default_factory=list, description="observed states (empty = all)")
    true_values: Dict[str, float] = Field(default_factory=dict, description="values used to generate the data")


class InverseRunRequest(FiniteMixin):
    system: SystemSpec
    time: TimeSpan = TimeSpan()
    unknowns: List[UnknownParameter] = Field(..., min_length=1)
    data_source: Literal["upload", "synthetic"]
    upload: Optional[UploadedObservations] = None
    synthetic: Optional[SyntheticObservations] = None
    reference_values: Dict[str, float] = Field(default_factory=dict,
                                               description="known true values of unknowns, for error reporting")
    options: InverseOptionsModel = InverseOptionsModel()

    @model_validator(mode="after")
    def _consistent(self):
        names = self.system.parameter_names
        for u in self.unknowns:
            if u.name not in names:
                raise ValueError(f"unknown parameter '{u.name}' is not a parameter of the system")
        if len({u.name for u in self.unknowns}) != len(self.unknowns):
            raise ValueError("a parameter is listed as unknown twice")
        if self.system.controls:
            raise ValueError("inverse training has no control inputs; use parameters instead")
        if self.data_source == "upload":
            if self.upload is None:
                raise ValueError("data_source 'upload' needs observations")
            states = self.system.state_names
            for col in self.upload.columns:
                if col not in states:
                    raise ValueError(f"observation column '{col}' is not a state variable")
            n = len(self.upload.t)
            for col, vals in self.upload.columns.items():
                if len(vals) != n:
                    raise ValueError(f"column '{col}' has {len(vals)} values but there are {n} times")
        elif self.synthetic is None:
            raise ValueError("data_source 'synthetic' needs generation settings")
        return self


def clean_observations(up: UploadedObservations, t_span: TimeSpan):
    """Drop rows with missing/non-finite values; check ordering and range.
    Returns (t, {state: values}, notes) or raises ValueError."""
    notes = []
    rows = []
    cols = list(up.columns)
    for i, t in enumerate(up.t):
        vals = [up.columns[c][i] for c in cols]
        if t is None or not math.isfinite(t) or any(v is None or not math.isfinite(v) for v in vals):
            continue
        rows.append((t, vals))
    dropped = len(up.t) - len(rows)
    if dropped:
        notes.append(f"{dropped} row(s) with missing or non-numeric values were dropped.")
    if len(rows) < 2:
        raise ValueError("fewer than 2 complete observation rows")
    ts = [r[0] for r in rows]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("observation times must be strictly increasing (sort the file and remove duplicates)")
    outside = sum(1 for t in ts if t < t_span.t0 or t > t_span.t1)
    if outside:
        raise ValueError(f"{outside} observation time(s) lie outside the time domain "
                         f"[{t_span.t0:g}, {t_span.t1:g}]")
    return ts, {c: [r[1][k] for r in rows] for k, c in enumerate(cols)}, notes
