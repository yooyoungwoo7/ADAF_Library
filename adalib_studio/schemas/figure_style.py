"""
Free-form figure style from the AI assistant: Plotly layout / trace patches per figure
(frontend: plots/figureStyle.ts applies them on top of the Paper/Screen look).

    {"<figure id>" | "prefix*" | "*": {
        "layout": {...Plotly layout, deep-merged; "xaxes"/"yaxes" reach every x/y axis...},
        "traces": {"<selector>": {...Plotly trace attributes}},
        "export": {"width_cm", "height_cm", "dpi", "format"}},
     "@<system name>": {same figure keys: patches for that problem only, applied after the above}}

Answers are JSON merge patches (RFC 7386): objects merge, null removes a key.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_JSON = 40_000
SELECTOR = re.compile(r"^(\*|#\d+|name:.+|legendgroup:.+|axis:[xy]\d*)$")


class FigureExport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width_cm: Optional[float] = Field(None, gt=0, le=100)
    height_cm: Optional[float] = Field(None, gt=0, le=100)
    dpi: Optional[float] = Field(None, ge=72, le=1200)
    format: Optional[Literal["png", "svg"]] = None


def merge_patch(target, patch):
    """RFC 7386 JSON merge patch; empty objects left behind are dropped."""
    if not isinstance(patch, dict):
        return patch
    out = dict(target) if isinstance(target, dict) else {}
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            merged = merge_patch(out.get(k), v)
            if isinstance(merged, dict) and not merged:
                out.pop(k, None)
            else:
                out[k] = merged
    return out


def _id_ok(key: str, ids: Iterable[str]) -> bool:
    if key == "*":
        return True
    if key.endswith("*"):
        return any(i.startswith(key[:-1]) for i in ids)
    return key in ids


def _check_figure(key: str, fig) -> None:
    if not isinstance(fig, dict):
        raise ValueError(f"figure_style.{key} must be an object with layout, traces and/or export.")
    extra = sorted(set(fig) - {"layout", "traces", "export"})
    if extra:
        raise ValueError(f"figure_style.{key}: unknown key(s) {extra}; use layout, traces, export.")
    if "layout" in fig and not isinstance(fig["layout"], dict):
        raise ValueError(f"figure_style.{key}.layout must be an object of Plotly layout attributes.")
    traces = fig.get("traces", {})
    if not isinstance(traces, dict):
        raise ValueError(f'figure_style.{key}.traces must map selectors ("*", "#0", "name:...", "axis:y2") '
                         "to trace attributes.")
    for sel, attrs in traces.items():
        if not SELECTOR.match(sel):
            raise ValueError(f"figure_style.{key}.traces: bad selector '{sel}'; use \"*\", \"#<index>\", "
                             "\"name:<trace name>\", \"legendgroup:<group>\" or \"axis:y2\".")
        if not isinstance(attrs, dict):
            raise ValueError(f"figure_style.{key}.traces['{sel}'] must be an object of trace attributes.")
    if "export" in fig:
        if not isinstance(fig["export"], dict):
            raise ValueError(f"figure_style.{key}.export must be an object.")
        try:
            FigureExport(**fig["export"])
        except ValidationError as exc:
            raise ValueError(f"figure_style.{key}.export: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))


def check_figure_style(style) -> dict:
    """Raise ValueError (message for the model) unless `style` is a valid figure style:
    {figure key: patch} for the whole module, plus {"@<system>": {figure key: patch}} per problem."""
    if not isinstance(style, dict):
        raise ValueError('"figure_style" must map figure ids to {layout, traces, export}.')
    text = json.dumps(style)
    if len(text) > MAX_JSON:
        raise ValueError(f"figure_style is too large ({len(text)} characters, max {MAX_JSON}).")
    if "javascript:" in text.lower() or "<script" in text.lower():
        raise ValueError("figure_style must not contain scripts.")
    for key, fig in style.items():
        if key.startswith("@"):
            if not isinstance(fig, dict):
                raise ValueError(f"figure_style.{key} must map figure ids to patches for that problem.")
            for k, f in fig.items():
                if k.startswith("@"):
                    raise ValueError(f"figure_style.{key}.{k}: problem sections cannot be nested.")
                _check_figure(f"{key}.{k}", f)
        else:
            _check_figure(key, fig)
    return style


def _check_ids(patch: dict, ids, scope: str = "") -> None:
    """Figures named by this answer must exist (styles kept from earlier problems are not re-checked)."""
    for key, v in patch.items():
        if v is None:
            continue
        if key.startswith("@") and not scope:
            if not isinstance(v, dict):
                raise ValueError(f"figure_style.{key} must map figure ids to patches for that problem.")
            _check_ids(v, ids, key)
        elif not _id_ok(key, ids):
            where = f"figure_style.{scope}" if scope else "figure_style"
            raise ValueError(f"{where}: unknown figure '{key}'; use \"*\", a prefix with * or one of {ids}.")


def merge_figure_style(current, patch, figure_ids) -> dict:
    if patch is None:
        return current or {}
    if not isinstance(patch, dict):
        raise ValueError('"figure_style" must be an object.')
    _check_ids(patch, list(figure_ids))
    return check_figure_style(merge_patch(current or {}, patch))
