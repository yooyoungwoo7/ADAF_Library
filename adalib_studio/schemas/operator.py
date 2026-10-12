from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from .common import FiniteMixin, SystemSpec

BuiltinOperatorSystem = Literal["cstr", "triple_tank", "fedbatch_bioreactor"]


class OperatorOptionsModel(FiniteMixin):
    """Subset of ``adalib.OperatorOptions`` (0.1.0) exposed in the Studio.
    Defaults equal the package's; the UI offers smaller presets because the
    built-in defaults train for tens of minutes or more."""

    n_train: int = Field(2000, ge=2, le=200_000)
    n_val: int = Field(200, ge=1, le=50_000)
    epochs: int = Field(1000, ge=1, le=100_000)
    batch_size: int = Field(256, ge=1, le=65_536)
    lr: float = Field(3e-3, gt=0, le=1.0)
    hidden: int = Field(128, ge=4, le=4096)
    n_layers: int = Field(3, ge=1, le=12)
    seed: int = 42
    basis: Literal["lpa", "adaf"] = "lpa"          # built-in path only
    use_lr_schedule: bool = True                   # built-in path only
    lpa_n_panels: int = Field(8, ge=1, le=64)      # generic path only
    lpa_max_order: int = Field(6, ge=1, le=20)     # generic path only
    lpa_nt_seg: int = Field(20, ge=4, le=400)      # generic path only


class OperatorTrainRequest(FiniteMixin):
    source: Literal["custom", "builtin"]
    # custom: generic physics-residual-only path (CallableODESystem)
    system: Optional[SystemSpec] = None
    n_seg: int = Field(25, ge=1, le=1000, description="custom: rollout segments")
    t_final: float = Field(1.0, gt=0, description="custom: rollout horizon; segment length = t_final / n_seg")
    # built-in: legacy problem with its own time grid, ranges and physics
    builtin: Optional[BuiltinOperatorSystem] = None
    options: OperatorOptionsModel = OperatorOptionsModel()
    label: str = ""

    @model_validator(mode="after")
    def _source(self):
        if self.source == "custom":
            if self.system is None:
                raise ValueError("source 'custom' needs a system definition")
            for s in self.system.states:
                if s.lower is None or s.upper is None or not s.upper > s.lower:
                    raise ValueError(f"state '{s.name}': give a training range with upper > lower")
        elif self.builtin is None:
            raise ValueError("source 'builtin' needs a built-in system name")
        if self.options.n_val > self.options.n_train:
            raise ValueError("n_val should not exceed n_train")
        return self


class OperatorInferRequest(FiniteMixin):
    model_id: str = Field(..., description="job id of a completed operator training run")
    x0: List[float]
    inputs: List[float] = Field(default_factory=list,
                                description="custom: control values u; built-in: legacy θ vector")
    reference: bool = True
