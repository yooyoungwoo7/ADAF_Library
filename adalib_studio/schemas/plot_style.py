"""Figure style for the Forward trajectory plots, set by the user or the AI assistant.

Every field is optional: unset fields keep the Paper/Screen look. Font sizes are in points,
export sizes in centimetres (converted for Plotly in the browser).
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Dash = Literal["solid", "dash", "dot", "dashdot", "longdash"]
LegendPosition = Literal["top", "top-left", "top-right", "bottom-left", "bottom-right", "right", "hidden"]

# hex, a CSS colour name, or rgb()/rgba()
_COLOR = r"^(#[0-9a-fA-F]{3,8}|[a-zA-Z]{3,20}|rgba?\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*(,\s*[0-9.]+\s*)?\))$"
Color = Field(None, pattern=_COLOR, max_length=40)


class PlotStyleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    font_family: Optional[str] = Field(None, max_length=80)
    font_size: Optional[float] = Field(None, ge=5, le=40)          # base text, pt
    title_font_size: Optional[float] = Field(None, ge=5, le=48)
    tick_font_size: Optional[float] = Field(None, ge=4, le=40)
    legend_font_size: Optional[float] = Field(None, ge=4, le=40)

    line_width: Optional[float] = Field(None, ge=0.2, le=10)       # ADA lines
    colors: Optional[Dict[str, str]] = None                        # state -> ADA line colour
    dashes: Optional[Dict[str, Dash]] = None                       # state -> ADA line dash
    show_reference: Optional[bool] = None
    ref_color: Optional[str] = Color
    ref_line_width: Optional[float] = Field(None, ge=0.2, le=10)
    ref_dash: Optional[Dash] = None

    title: Optional[str] = Field(None, max_length=120)             # figure title (one plot)
    x_title: Optional[str] = Field(None, max_length=80)
    y_title: Optional[str] = Field(None, max_length=80)            # one plot
    y_titles: Optional[Dict[str, str]] = None                      # per-state panels: state -> y label
    panel_titles: Optional[Dict[str, str]] = None                  # per-state panels: state -> title
    x_range: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    y_range: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    log_y: Optional[bool] = None
    grid: Optional[bool] = None

    legend_position: Optional[LegendPosition] = None
    legend_box: Optional[bool] = None
    ada_label: Optional[str] = Field(None, max_length=60)          # "ADA" in the legend
    ref_label: Optional[str] = Field(None, max_length=60)          # "reference, BDF"

    export_width_cm: Optional[float] = Field(None, ge=2, le=60)
    export_height_cm: Optional[float] = Field(None, ge=2, le=60)
    export_dpi: Optional[int] = Field(None, ge=72, le=1200)
    export_format: Optional[Literal["png", "svg"]] = None

    @field_validator("colors")
    @classmethod
    def _colors(cls, v):
        import re
        for state, c in (v or {}).items():
            if not re.match(_COLOR, c or ""):
                raise ValueError(f"colour for '{state}' must be hex (#d62728), a CSS name (red) or rgb(...), got {c!r}")
        return v

    @field_validator("x_range", "y_range")
    @classmethod
    def _range(cls, v):
        if v is not None and not v[1] > v[0]:
            raise ValueError("range must be [low, high] with high > low")
        return v


PER_STATE_KEYS = ("colors", "dashes", "y_titles", "panel_titles")


def merge_style(current: Optional[dict], patch: dict, state_names: List[str]) -> dict:
    """Apply a style patch: a key set to null resets it; per-state maps merge by state
    (a state set to null resets that state). Raises ValueError for unknown states."""
    out = dict(current or {})
    for key, value in patch.items():
        if key in PER_STATE_KEYS and isinstance(value, dict):
            unknown = sorted(set(value) - set(state_names))
            if unknown:
                raise ValueError(f"plot_style.{key}: unknown state(s) {unknown}; states are {state_names}")
            merged = dict(out.get(key) or {})
            for state, v in value.items():
                if v is None:
                    merged.pop(state, None)
                else:
                    merged[state] = v
            value = merged or None
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return PlotStyleModel(**out).model_dump(exclude_none=True)
