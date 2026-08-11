"""
adalib/utils/paths.py
Path utilities for locating the vendored legacy backend.

The legacy backend lives at adalib/_vendor/legacy/ so it is co-located
with the installed package and works after both `pip install -e .` and
`pip install adalib-ode` from a wheel.
"""
from __future__ import annotations
import os
import sys
import pathlib
from contextlib import contextmanager


def _adalib_pkg() -> pathlib.Path:
    """Root of the installed adalib package (the directory that contains utils/)."""
    # __file__ = adalib/utils/paths.py  →  parent = adalib/utils/  →  parent = adalib/
    return pathlib.Path(__file__).parent.parent.resolve()


def package_dir() -> pathlib.Path:
    """Root of the installed ``adalib`` package directory."""
    return _adalib_pkg()


def package_root() -> pathlib.Path:
    """Alias for package_dir() — kept for backward compatibility."""
    return _adalib_pkg()


def project_root() -> pathlib.Path:
    """Parent of the source checkout (one level above ``adalib/``).

    .. warning::
        This path only exists in editable / source-tree layouts.
        Do not rely on it in installed packages.
    """
    return _adalib_pkg().parent


def vendor_legacy_root() -> pathlib.Path:
    """Root of the vendored legacy backend: ``adalib/_vendor/legacy/``."""
    p = _adalib_pkg() / "_vendor" / "legacy"
    if not p.exists():
        raise RuntimeError(
            "ADALib legacy backend was not found at expected path:\n"
            f"  {p}\n"
            "This usually means the package was installed from a wheel that was "
            "built without vendored legacy files.\n"
            "Re-install from source with:  pip install -e .\n"
            "or rebuild the wheel after verifying adalib/_vendor/legacy/ is present."
        )
    return p


def legacy_forward_root() -> pathlib.Path:
    """``adalib/_vendor/legacy/forward_problem_original/``"""
    return vendor_legacy_root() / "forward_problem_original"


def legacy_operator_root() -> pathlib.Path:
    """``adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/``"""
    return vendor_legacy_root() / "operator_mpc_original" / "cstr_mpc_op"


def data_dir(problem_name: str | None = None) -> pathlib.Path:
    """``…/cstr_mpc_op/data_files/`` or ``…/data_files/<problem_name>/``."""
    base = legacy_operator_root() / "data_files"
    return base / problem_name if problem_name else base


def results_dir(problem_name: str | None = None) -> pathlib.Path:
    """``…/cstr_mpc_op/results/`` or ``…/results/<problem_name>/``."""
    base = legacy_operator_root() / "results"
    return base / problem_name if problem_name else base


def checkpoints_dir(problem_name: str | None = None) -> pathlib.Path:
    """Alias for :func:`results_dir` — checkpoints live under ``results/``."""
    return results_dir(problem_name)


def ensure_legacy_on_path() -> None:
    """Insert legacy roots into ``sys.path`` (idempotent)."""
    fwd = str(legacy_forward_root())
    op  = str(legacy_operator_root())
    if fwd not in sys.path:
        sys.path.insert(0, fwd)
    if op not in sys.path:
        sys.path.insert(0, op)


@contextmanager
def temporary_working_directory(path):
    """Context manager: ``cd`` to *path*, restore original ``cwd`` on exit."""
    orig = os.getcwd()
    try:
        os.chdir(path)
        yield pathlib.Path(path)
    finally:
        os.chdir(orig)
