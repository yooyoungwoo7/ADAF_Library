"""
Free-form figure style from the AI assistant: Plotly layout / trace patches per figure
(frontend: plots/figureStyle.ts applies them on top of the Paper/Screen look).

    {"<figure id>" | "prefix*" | "*": {
        "layout": {...Plotly layout, deep-merged; "xaxes"/"yaxes" reach every x/y axis...},
        "traces": {"<selector>": {...Plotly trace attributes}},
        "export": {"width_cm", "height_cm", "dpi", "format"}}}

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


def check_figure_style(style, figure_ids: Iterable[str], new_keys: Optional[Iterable[str]] = None) -> dict:
    """Raise ValueError (message for the model) unless `style` is a valid figure style. Only `new_keys`
    (default: all) must name a known figure, so a style kept from an earlier system stays valid."""
    ids = list(figure_ids)
    new = set(style if new_keys is None else new_keys)
    if not isinstance(style, dict):
        raise ValueError('"figure_style" must map figure ids to {layout, traces, export}.')
    text = json.dumps(style)
    if len(text) > MAX_JSON:
        raise ValueError(f"figure_style is too large ({len(text)} characters, max {MAX_JSON}).")
    if "javascript:" in text.lower() or "<script" in text.lower():
        raise ValueError("figure_style must not contain scripts.")
    for key, fig in style.items():
        if key in new and not _id_ok(key, ids):
            raise ValueError(f"figure_style: unknown figure '{key}'; use \"*\", a prefix with * or one of {ids}.")
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
            try:
                FigureExport(**fig["export"]) if isinstance(fig["export"], dict) else FigureExport.model_validate(fig["export"])
            except ValidationError as exc:
                raise ValueError(f"figure_style.{key}.export: " + "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
    return style


def merge_figure_style(current, patch, figure_ids) -> dict:
    if patch is None:
        return current or {}
    if not isinstance(patch, dict):
        raise ValueError('"figure_style" must be an object.')
    return check_figure_style(merge_patch(current or {}, patch), figure_ids,
                              new_keys=[k for k, v in patch.items() if v is not None])
