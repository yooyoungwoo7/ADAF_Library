from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import FiniteMixin, SystemSpec, TimeSpan


class ForwardOptionsModel(FiniteMixin):
    """Mirror of ``adalib.ForwardOptions`` (0.1.0). Defaults equal the installed
    package's (asserted in tests/test_package.py). ``verbose`` is not exposed:
    workers always run verbose so the job console can show progress."""

    basis: Literal["adaf", "lpa"] = "adaf"
    n_seg: int = Field(50, ge=1, le=2000)
    N_p: int = Field(5, ge=1, le=500)
    N_m: int = Field(100, ge=1, le=2000)
    Nt_total: int = Field(2500, ge=2, le=200_000)
    epochs: int = Field(5, ge=0, le=10_000)
    adam_inner: int = Field(100, ge=1, le=100_000)
    adam_lr: float = Field(1e-3, gt=0, le=1.0)
    use_lbfgs: bool = True
    dtype: Literal["float32", "float64"] = "float64"
    gamma: float = Field(0.8, gt=0)
    L: float = 1.0
    order: int = Field(3, ge=1, le=50)

    @model_validator(mode="after")
    def _grid(self):
        if self.basis == "adaf" and self.Nt_total % self.n_seg != 0:
            raise ValueError(f"Nt_total ({self.Nt_total}) must be divisible by n_seg ({self.n_seg}) "
                             "for the ADA-F basis")
        return self


class ForwardRunRequest(FiniteMixin):
    system: SystemSpec
    time: TimeSpan = TimeSpan()
    options: ForwardOptionsModel = ForwardOptionsModel()
    reference: bool = Field(True, description="Also compute adalib's scipy BDF reference "
                                              "(adalib.utils.reference.solve_reference_ivp) for comparison")
