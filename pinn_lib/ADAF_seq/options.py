from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import tensorflow as tf
import numpy as np


@dataclass
class GridOptions:
    lb: float = 0.0
    ub: float = 1.0
    Nt_total: int = 2000
    n_seg: int = 10
    Nt_seg: Optional[int] = None
    gamma: float = 0.8
    L: float = 1.0
    N_p: int = 100
    N_m: int = 100
    

    def T(self) -> float:
        return float(self.ub - self.lb)


@dataclass
class AdamOptions:
    epochs: int = 20
    inner: int = 50
    lr: float = 1e-3
    seed: int = 0
    dtype: Union[str, tf.DType] = "float32"
    xla_predict: bool = True
    xla_step: bool = False


@dataclass
class LBFGSOptions:
    use: bool = True
    method: str = "L-BFGS-B"
    options: Dict[str, Any] = field(default_factory=lambda: {
        "maxiter": 500,
        "maxfun": 50000,
        "maxcor": 50,
        "maxls": 50,
        "ftol": np.finfo(float).eps,
        "gtol": np.finfo(float).eps,
        "iprint": -1,
    })