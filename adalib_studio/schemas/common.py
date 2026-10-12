from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StateSpec(BaseModel):
    name: str
    initial: float = Field(..., description="Initial condition x(t0)")
    unit: Optional[str] = None
    # training domain (Operator / MPC); ignored by the Forward solver
    lower: Optional[float] = None
    upper: Optional[float] = None


class ControlSpec(BaseModel):
    """External input u held constant over each segment (Operator / MPC)."""
    name: str
    value: float = 0.0
    lower: float
    upper: float

    @model_validator(mode="after")
    def _ordered(self):
        if not self.upper > self.lower:
            raise ValueError(f"control '{self.name}': upper bound must exceed lower bound")
        return self


class ParameterSpec(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None


class SystemSpec(BaseModel):
    """A user-defined autonomous ODE system dx_i/dt = f_i(x; p)."""

    name: str = "custom_system"
    states: List[StateSpec] = Field(..., min_length=1, max_length=20)
    parameters: List[ParameterSpec] = Field(default_factory=list, max_length=50)
    equations: List[str] = Field(..., min_length=1, max_length=20,
                                 description="equations[i] is the right-hand side of d(states[i])/dt")
    controls: List[ControlSpec] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _counts_match(self):
        if len(self.equations) != len(self.states):
            raise ValueError(f"{len(self.states)} state(s) but {len(self.equations)} equation(s)")
        return self

    @property
    def state_names(self) -> List[str]:
        return [s.name.strip() for s in self.states]

    @property
    def parameter_names(self) -> List[str]:
        return [p.name.strip() for p in self.parameters]

    @property
    def control_names(self) -> List[str]:
        return [c.name.strip() for c in self.controls]


class TimeSpan(BaseModel):
    t0: float = 0.0
    t1: float = 1.0

    @model_validator(mode="after")
    def _ordered(self):
        if not self.t1 > self.t0:
            raise ValueError("t_end must be greater than t_start")
        return self


class EquationIssue(BaseModel):
    message: str
    index: Optional[int] = None
    field: Optional[str] = None


class ValidateResponse(BaseModel):
    ok: bool
    errors: List[EquationIssue] = []
    latex: List[str] = []
    symbols: List[List[str]] = []
    unused_parameters: List[str] = []


class JobCreated(BaseModel):
    job_id: str
    status: str


def _finite(v: float) -> float:
    import math
    if not math.isfinite(v):
        raise ValueError("must be a finite number")
    return v


class FiniteMixin(BaseModel):
    @field_validator("*", mode="after")
    @classmethod
    def _check_finite(cls, v):
        if isinstance(v, float):
            return _finite(v)
        return v
