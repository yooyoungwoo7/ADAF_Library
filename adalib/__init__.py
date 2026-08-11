"""
adalib — Unified ADA-based ODE Library
=======================================
Three integrated features:

  - Forward  : ForwardSolver, ForwardOptions, run_forward
               Generic CallableODESystem fully supported.
  - Operator : OperatorLearner, run_operator
               Built-in systems only (cstr, triple_tank, fedbatch_bioreactor, …).
  - MPC      : run_mpc
               Built-in systems (cstr, triple_tank, fedbatch_bioreactor) and
               user-defined CallableODESystem — uses ADA LPA Operator NN as
               surrogate (same OperatorNet/BatchLPABasis as built-in systems).

Generic Operator learning via run_operator(CallableODESystem, …) is planned
for a future release.
"""
__version__ = "0.1.0"

# -- ODE systems ----------------------------------------------------------
from .systems import (
    ODESystem, FedBatchBioreactor, CSTR,
    TripleTank, LotkaVolterra, EulerRigidBody,
)
from .systems.callable_system import CallableODESystem
from .systems.registry import get_system, list_systems, register_system

# -- Forward solver -------------------------------------------------------
from .forward import ForwardSolver
from .forward.options import ForwardOptions

# -- Operator learning (low-level) ----------------------------------------
from .operator import predict_step, predict_rollout, OperatorLearner
from .operator.options import OperatorOptions

# -- MPC options ----------------------------------------------------------
from .mpc.options import MPCOptions

# -- High-level workflow API ----------------------------------------------
from .workflows import run_forward, run_operator, run_mpc, run_inverse, run
from .workflows.operator_workflow import OperatorResult
from .workflows.mpc_workflow      import MPCResult

# -- Inverse training API -------------------------------------------------
from .inverse import (
    ObservationData,
    InverseParameter,
    InverseOptions,
    InverseResult,
    data_gen,
)
from .workflows.forward_workflow import ForwardResult

# -- Utils subpackage (ensure adalib.utils.xxx always works) --------------
from . import utils

__all__ = [
    "__version__",
    # systems
    "ODESystem", "CallableODESystem",
    "FedBatchBioreactor", "CSTR", "TripleTank", "LotkaVolterra", "EulerRigidBody",
    "get_system", "list_systems", "register_system",
    # forward
    "ForwardSolver", "ForwardOptions",
    # operator (low-level)
    "OperatorLearner", "predict_step", "predict_rollout",
    # options
    "ForwardOptions", "OperatorOptions", "MPCOptions",
    # high-level workflows
    "run_forward", "run_operator", "run_mpc", "run_inverse", "run",
    # result types
    "ForwardResult", "OperatorResult", "MPCResult", "InverseResult",
    # inverse API
    "ObservationData", "InverseParameter", "InverseOptions",
    "data_gen",
]
