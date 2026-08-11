"""
adalib/utils/legacy_context.py
Utilities for temporarily configuring the legacy operator package.

The legacy package reads PROBLEM and BASIS from environment variables at
module import / reload time.  _reload_legacy_chain() re-executes all
config-dependent legacy modules in dependency order so that the correct
problem-specific constants (N_SEG, T_FINAL, INPUT_DIM, RES_SCALE, …) are
active for the lifetime of a run_operator / run_mpc call.
"""
from __future__ import annotations

import os
import sys
import importlib
from contextlib import contextmanager

from .paths import legacy_operator_root


# Modules that read from config at import/reload time, in dependency order.
_LEGACY_CONFIG_CHAIN = [
    "config",
    "utils.io_utils",
    "problems.base_problem",
    "problems.lotka_problem",
    "problems.bioreactor_problem",
    "problems.cstr_problem",
    "problems.triple_tank_problem",
    "problems.cstr_mpc_problem",
    "problems.triple_tank_mpc_problem",
    "problems.registry",
    "data.dataset_builder",
    "models.lpa_basis",
    "models.adaf_basis",
    "models.basis",
    "models.operator_net",
    "models.learner",
]


def _ensure_legacy_path() -> None:
    op_path = str(legacy_operator_root())
    if op_path not in sys.path:
        sys.path.insert(0, op_path)


def reload_legacy_chain(problem: str, basis: str = "lpa") -> None:
    """Set PROBLEM/BASIS env vars and reload all config-dependent legacy modules.

    Must be called before creating OperatorLearner or calling dataset builders
    for any problem other than the default (lotka).  Safe to call multiple
    times; subsequent calls with a different problem simply re-configure.
    """
    _ensure_legacy_path()

    os.environ["PROBLEM"] = problem
    os.environ["BASIS"] = basis
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import logging as _logging
    _log = _logging.getLogger(__name__)

    for mod_name in _LEGACY_CONFIG_CHAIN:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception as _e:
                _log.warning(
                    "reload_legacy_chain: failed to reload '%s' "
                    "(PROBLEM=%s, BASIS=%s): %s",
                    mod_name, problem, basis, _e,
                )
        else:
            try:
                importlib.import_module(mod_name)
            except ModuleNotFoundError:
                pass  # optional module (e.g. triple_tank_mpc_problem may be absent)
            except Exception as _e:
                _log.warning(
                    "reload_legacy_chain: failed to import '%s' "
                    "(PROBLEM=%s, BASIS=%s): %s",
                    mod_name, problem, basis, _e,
                )


@contextmanager
def legacy_problem_context(problem: str, basis: str = "lpa"):
    """Context manager: set legacy PROBLEM/BASIS, reload modules, restore on exit.

    Usage::

        with legacy_problem_context("cstr_mpc", basis="lpa") as mods:
            cfg = mods["config"]
            db  = mods["dataset_builder"]
            ml  = mods["learner_mod"]
            preg = mods["registry"]
    """
    _ensure_legacy_path()

    old_problem = os.environ.get("PROBLEM")
    old_basis   = os.environ.get("BASIS")

    try:
        reload_legacy_chain(problem, basis)

        import config as _cfg
        import data.dataset_builder as _db
        import models.learner as _ml
        import problems.registry as _preg

        yield {
            "config":         _cfg,
            "dataset_builder": _db,
            "learner_mod":    _ml,
            "registry":       _preg,
        }
    finally:
        if old_problem is None:
            os.environ.pop("PROBLEM", None)
        else:
            os.environ["PROBLEM"] = old_problem
        if old_basis is None:
            os.environ.pop("BASIS", None)
        else:
            os.environ["BASIS"] = old_basis
