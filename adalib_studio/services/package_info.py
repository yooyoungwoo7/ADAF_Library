"""
Package metadata and option defaults read from the installed adalib-IFL
*without importing adalib* (importing it loads TensorFlow and adalib's
process-global state, which must stay inside worker processes).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import ADALIB_DISTRIBUTION, SUPPORTED_ADALIB_VERSIONS


def adalib_version() -> Optional[str]:
    try:
        return metadata.version(ADALIB_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def dependency_versions() -> Dict[str, Optional[str]]:
    out = {}
    for dist in ("tensorflow", "numpy", "scipy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def version_supported() -> bool:
    return adalib_version() in SUPPORTED_ADALIB_VERSIONS


def _package_dir() -> Optional[Path]:
    spec = importlib.util.find_spec("adalib")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _load_isolated(relpath: str, name: str):
    """Load one dependency-free module file of adalib by path."""
    root = _package_dir()
    if root is None:
        raise RuntimeError("adalib is not installed in the backend environment.")
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve annotations via sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@lru_cache(maxsize=1)
def forward_option_defaults() -> Dict[str, Any]:
    """Field defaults of ``adalib.forward.options.ForwardOptions``."""
    mod = _load_isolated("forward/options.py", "_adalib_forward_options")
    return {f.name: f.default for f in dataclasses.fields(mod.ForwardOptions)}
