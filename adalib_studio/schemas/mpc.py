from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from .common import FiniteMixin, SystemSpec

BuiltinMPCSystem = Literal["cstr", "triple_tank", "fedbatch_bioreactor"]


class MPCOptionsModel(FiniteMixin):
    """Subset of ``adalib.MPCOptions`` (0.1.0); defaults equal the package's.
    Every run trains its own surrogate operator before the closed loop."""

    n_steps: int = Field(20, ge=1, le=10_000)
    horizon: Optional[int] = Field(None, ge=1, le=200)
    gradient: Optional[Literal["autodiff", "fd"]] = None
    optimizer: Literal["SLSQP", "CEM", "MPPI"] = "SLSQP"      # CEM/MPPI: built-in systems only
    n_train: int = Field(200, ge=2, le=200_000)
    n_val: int = Field(50, ge=1, le=50_000)
    epochs: int = Field(300, ge=1, le=100_000)
    batch_size: int = Field(256, ge=1, le=65_536)
    lr: float = Field(3e-3, gt=0, le=1.0)
    hidden: int = Field(256, ge=4, le=4096)
    n_layers: int = Field(4, ge=1, le=12)
    seed: int = 42
    basis: Literal["lpa", "adaf"] = "lpa"                     # built-in surrogate
    use_lr_schedule: bool = True
    plant_solver: str = "RK45"                                # custom path plant
    lpa_n_panels: int = Field(8, ge=1, le=64)
    lpa_max_order: int = Field(6, ge=1, le=20)
    lpa_nt_seg: int = Field(20, ge=4, le=400)


class MPCRunRequest(FiniteMixin):
    source: Literal["custom", "builtin"]
    # custom: generic tracking MPC on a CallableODESystem
    system: Optional[SystemSpec] = None
    dt: float = Field(0.1, gt=0, description="custom: control interval")
    targets: Dict[str, float] = Field(default_factory=dict, description="tracked state -> reference value")
    tracking_weights: Dict[str, float] = Field(default_factory=dict)
    control_weights: Dict[str, float] = Field(default_factory=dict)
    # built-in: legacy MPC problems (cstr / triple_tank tracking, bioreactor economic)
    builtin: Optional[BuiltinMPCSystem] = None
    x0: List[float] = Field(default_factory=list)
    control_bounds: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    options: MPCOptionsModel = MPCOptionsModel()

    @model_validator(mode="after")
    def _consistent(self):
        for name, (lo, hi) in self.control_bounds.items():
            if not hi > lo:
                raise ValueError(f"control '{name}': upper bound must exceed lower bound")
        if self.source == "custom":
            s = self.system
            if s is None:
                raise ValueError("source 'custom' needs a system definition")
            if not s.controls:
                raise ValueError("MPC needs at least one control input")
            if not self.targets:
                raise ValueError("choose at least one state to track and its reference value")
            for n in self.targets:
                if n not in s.state_names:
                    raise ValueError(f"tracked variable '{n}' is not a state")
            for st in s.states:
                if st.lower is None or st.upper is None or not st.upper > st.lower:
                    raise ValueError(f"state '{st.name}': give a range with upper > lower (surrogate training box)")
            if self.options.optimizer != "SLSQP":
                raise ValueError("custom systems use SLSQP in adalib 0.1.0 (CEM/MPPI are built-in only)")
        else:
            if self.builtin is None:
                raise ValueError("source 'builtin' needs a built-in system")
            n = {"cstr": 4, "triple_tank": 3, "fedbatch_bioreactor": 4}[self.builtin]
            if len(self.x0) != n:
                raise ValueError(f"{self.builtin} needs {n} initial values")
        return self


# ── split workflow: train a surrogate once, run the closed loop many times ──

class MPCSurrogateOptions(FiniteMixin):
    """Surrogate-training subset of ``adalib.MPCOptions`` (defaults equal the package's)."""

    n_train: int = Field(200, ge=2, le=200_000)
    n_val: int = Field(50, ge=1, le=50_000)
    epochs: int = Field(300, ge=1, le=100_000)
    batch_size: int = Field(256, ge=1, le=65_536)
    lr: float = Field(3e-3, gt=0, le=1.0)
    hidden: int = Field(256, ge=4, le=4096)
    n_layers: int = Field(4, ge=1, le=12)
    seed: int = 42
    basis: Literal["lpa", "adaf"] = "lpa"                     # built-in surrogate
    use_lr_schedule: bool = True
    lpa_n_panels: int = Field(8, ge=1, le=64)                 # custom (generic LPA surrogate)
    lpa_max_order: int = Field(6, ge=1, le=20)
    lpa_nt_seg: int = Field(20, ge=4, le=400)

    @model_validator(mode="after")
    def _sizes(self):
        if self.n_val > self.n_train:
            raise ValueError("validation samples must not exceed training samples")
        return self


class MPCTrainRequest(FiniteMixin):
    """Train the MPC surrogate only (``MPCOptions(run_closed_loop=False)``)."""

    source: Literal["custom", "builtin"]
    system: Optional[SystemSpec] = None
    dt: float = Field(0.1, gt=0, description="custom: control interval; the surrogate predicts one dt ahead")
    builtin: Optional[BuiltinMPCSystem] = None
    label: Optional[str] = Field(None, max_length=120)
    options: MPCSurrogateOptions = MPCSurrogateOptions()

    @model_validator(mode="after")
    def _consistent(self):
        if self.source == "custom":
            s = self.system
            if s is None:
                raise ValueError("source 'custom' needs a system definition")
            if not s.controls:
                raise ValueError("MPC needs at least one control input")
            for st in s.states:
                if st.lower is None or st.upper is None or not st.upper > st.lower:
                    raise ValueError(f"state '{st.name}': give a range with upper > lower (surrogate training box)")
        elif self.builtin is None:
            raise ValueError("source 'builtin' needs a built-in system")
        return self


class MPCController(FiniteMixin):
    """One controller on the shared surrogate: SLSQP with FD or autodiff gradients, or CEM / MPPI sampling."""

    gradient: Optional[Literal["autodiff", "fd"]] = None
    optimizer: Literal["SLSQP", "CEM", "MPPI"] = "SLSQP"

    @property
    def label(self) -> str:
        if self.optimizer != "SLSQP":
            return self.optimizer
        return {"fd": "FD", "autodiff": "AD"}.get(self.gradient or "", "SLSQP")


class MPCLoopRequest(FiniteMixin):
    """Closed loop on a trained surrogate (``reuse_existing_operator=True``); no training."""

    model_id: str = Field(..., min_length=1, max_length=64)
    x0: List[float] = Field(default_factory=list)
    targets: Dict[str, float] = Field(default_factory=dict, description="tracked state -> reference value")
    tracking_weights: Dict[str, float] = Field(default_factory=dict)
    control_weights: Dict[str, float] = Field(default_factory=dict)
    control_bounds: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="built-in only")
    n_steps: int = Field(20, ge=1, le=10_000)
    horizon: Optional[int] = Field(None, ge=1, le=200)
    plant_solver: str = "RK45"                                # custom path plant
    controllers: List[MPCController] = Field(default_factory=lambda: [MPCController(gradient="autodiff")],
                                             min_length=1, max_length=4)

    @model_validator(mode="after")
    def _consistent(self):
        for name, (lo, hi) in self.control_bounds.items():
            if not hi > lo:
                raise ValueError(f"control '{name}': upper bound must exceed lower bound")
        labels = [c.label for c in self.controllers]
        if len(set(labels)) != len(labels):
            raise ValueError("each controller may be selected once")
        return self
