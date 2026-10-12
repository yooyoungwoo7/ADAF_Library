"""
Natural-language assistant for every Studio module (Forward, Operator, Inverse, MPC).

The user's sentence (plus what is on screen and the chat so far) goes to an
LLM (Anthropic Claude or OpenAI) that answers with JSON: the module's setup, or
only the parts to change. The answer is merged with what is on screen and
checked with the same parser and request models a run uses; if it fails, the
error is sent back once or twice for a corrected answer. The result only fills
the module's panels: the user still reviews it and presses Run (or Train).

API keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY, or from the settings
file (written from the UI). Keys are never sent to the browser. Plain HTTP via
urllib, so no SDK dependency. ANTHROPIC_BASE_URL / OPENAI_BASE_URL override the
endpoints (proxies, or any OpenAI-compatible server such as a local model).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

from ..config import SETTINGS_FILE
from ..schemas.assist import (AssistForwardRequest, AssistForwardResponse, AssistModuleRequest,
                              AssistModuleResponse, AssistProviderStatus, AssistSettingsUpdate, AssistStatus)
from ..schemas.common import SystemSpec, TimeSpan
from ..schemas.forward import ForwardOptionsModel
from ..schemas.inverse import InverseOptionsModel, SyntheticObservations, UnknownParameter
from ..schemas.mpc import MPCOptionsModel, MPCSurrogateOptions, MPCTrainRequest
from ..schemas.operator import OperatorOptionsModel, OperatorTrainRequest
from ..schemas.figure_style import merge_figure_style
from ..schemas.plot_style import PlotStyleModel, merge_style
from .equation_parser import CONSTANTS, FUNCTIONS
from .system_builder import validate_spec

PROVIDERS: Dict[str, Dict[str, str]] = {
    "anthropic": {"label": "Anthropic (Claude)", "env": "ANTHROPIC_API_KEY", "default_model": "claude-haiku-5-5"},
    "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "default_model": "gpt-5.4-mini"},
}
MAX_ATTEMPTS = 3
TIMEOUT_S = 90
BUILTINS = ("cstr", "triple_tank", "fedbatch_bioreactor")
MPC_CONTROLLERS = ("fd", "autodiff", "CEM", "MPPI")


class AssistError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ── settings ─────────────────────────────────────────────────────────────

def _load(path: Path = None) -> dict:
    path = path or SETTINGS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict, path: Path = None):
    path = path or SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _key(provider: str, settings: dict) -> Tuple[Optional[str], Optional[str]]:
    env = os.environ.get(PROVIDERS[provider]["env"], "").strip()
    if env:
        return env, "env"
    stored = (settings.get("llm", {}).get(provider, {}) or {}).get("api_key", "").strip()
    return (stored, "settings") if stored else (None, None)


def _model(provider: str, settings: dict) -> str:
    return (settings.get("llm", {}).get(provider, {}) or {}).get("model") or PROVIDERS[provider]["default_model"]


def status() -> AssistStatus:
    settings = _load()
    providers = {}
    for name, info in PROVIDERS.items():
        key, source = _key(name, settings)
        providers[name] = AssistProviderStatus(
            label=info["label"], configured=key is not None, source=source,
            key_hint=f"…{key[-4:]}" if key and len(key) > 8 else None,
            model=_model(name, settings), default_model=info["default_model"])
    chosen = settings.get("llm", {}).get("provider")
    if chosen not in PROVIDERS or not providers[chosen].configured:
        chosen = next((n for n, p in providers.items() if p.configured), None)
    return AssistStatus(provider=chosen, providers=providers, settings_file=str(SETTINGS_FILE))


def update_settings(upd: AssistSettingsUpdate) -> AssistStatus:
    settings = _load()
    llm = settings.setdefault("llm", {})
    entry = llm.setdefault(upd.provider, {})
    if upd.api_key is not None:
        if upd.api_key.strip():
            entry["api_key"] = upd.api_key.strip()
        else:
            entry.pop("api_key", None)
    if upd.model is not None:
        if upd.model.strip():
            entry["model"] = upd.model.strip()
        else:
            entry.pop("model", None)
    if upd.make_default:
        llm["provider"] = upd.provider
    _save(settings)
    return status()


# ── prompts ──────────────────────────────────────────────────────────────

INTRO = """You are the assistant inside ADALib Studio, the browser workspace of adalib (PyPI package
"adalib-IFL", the Anti-Derivative Approximator library). adalib solves and learns ODE systems in a
physics-informed way: it fits an ADA basis (ADA-F: adaptive Fourier on sequential time segments, or LPA:
Legendre polynomials) so that the ODE residual is minimised. Studio has four modules: Forward Solver,
Operator Learning, Inverse Problem and MPC.
The users are engineers and researchers who already know they are using adalib; do not explain it.
"""

SYSTEM_RULES = f"""Users often just name a system ("spring-damper", "Lotka-Volterra", "Duffing", "SIR model", "CSTR"),
possibly in Korean. Then infer the standard model and choose, without asking questions:
- the usual state variables and parameter names (e.g. x, v and m, c, k for a mass-spring-damper),
- textbook parameter values and a typical non-trivial initial condition,
- a time span that shows the characteristic behaviour (a few oscillation periods, the settling time,
  one epidemic wave), not much longer,
and mention these choices briefly in "reply".

System rules:
- equations[i] is the right-hand side of d(states[i])/dt, same order and count as states.
- Rewrite higher-order ODEs as first-order systems (e.g. theta'' -> states theta, omega with theta' = omega).
- Python expression syntax: + - * / and ** for powers (never ^), numbers, state, parameter (and control)
  names, functions {", ".join(sorted(FUNCTIONS))}, constants {", ".join(sorted(CONSTANTS))}.
- The time variable t must NOT appear (only autonomous systems); forcing that depends on t is not possible.
- Names: ASCII letters, digits and _, not starting with a digit or _, unique, and none of:
  {", ".join(sorted(FUNCTIONS))}, {", ".join(sorted(CONSTANTS))}, t.
- Put every physical constant in "parameters" (do not hard-code it in the equations) so the user can edit it.
- Values the user did not give: choose standard textbook values and say so in "reply".
- At most 20 states and 50 parameters; t1 > t0.
- If an image is attached, it shows the model (typeset or handwritten equations, a screenshot of a paper):
  transcribe it faithfully. Keep its symbols as names where valid (Greek letters spelled out: alpha, omega),
  take parameter values and initial conditions from it, and choose only what it leaves out (say so).
  Text with the image refines it ("only the first two equations", "use k = 3").
- If the user names or describes a different system, or asks for a new one ("new", "make", "set up",
  "define", "만들어", "구성해"), replace the one on screen. Only a message that asks for nothing
  (a greeting, a question about the current setup) leaves it unchanged; if in doubt, make the change.
- If the user asks to modify the current system (a value, a term, the time span), change only that and
  return the complete updated system.
"""

FIGURE_RULES = """
Figure style ("figure_style"): full Plotly.js control over every figure of this module (any layout or trace
attribute), applied at once without re-running and saved with the project. The figures on screen are listed
below with their current layout and traces, plus every figure id of the module. Send a JSON merge patch:
include only what changes; objects merge, null removes something set earlier (restoring the default look).
  "figure_style": {
    "<figure id>" | "<prefix>*" | "*" (every figure of this module): {
      "layout": {Plotly layout attributes, e.g. "xaxis": {"ticks": ""}, "legend": {...}, "font": {...},
                 "title": {"text": "..."}, "margin": {...}, "annotations": [...]; "xaxes" / "yaxes" apply
                 to every x / y axis of the figure (subplots have xaxis2, yaxis2, ...)},
      "traces": {"<selector>": {Plotly trace attributes, e.g. "line": {"color": "red", "width": 2, "dash": "dot"},
                 "marker": {...}, "name": "...", "showlegend": false, "visible": false}},
      "export": {"width_cm": 8, "height_cm": 6, "dpi": 600, "format": "png" | "svg"}
    }
  }
Trace selectors: "*" all, "#2" by index, "name:<trace name>", "legendgroup:<group>", "axis:y2" (every trace
drawn on that axis, i.e. one subplot). Look at the listed traces and axes to pick them; for subplots, a
state's panel is the axis whose title is that state.
Useful recipes: minor (small inner) ticks off: "yaxes"/"xaxes" {"minor": {"ticks": ""}}; all ticks off:
{"ticks": "", "minor": {"ticks": ""}}; ticks outside: {"ticks": "outside"}; legend under the plot:
"legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.3, "yanchor": "top"} (below the x-axis title; Plotly makes
room); legend inside top-right: {"x": 0.98, "xanchor": "right", "y": 0.98, "yanchor": "top"}; hide the
legend: "showlegend": false; grid: {"showgrid": true, "gridcolor": "#ddd"}; frame: {"mirror": true,
"showline": true}; log axis: {"type": "log"}; axis title: {"title": {"text": "Time [s]"}}.
Names that change with the problem: text may contain {system} (the current system's name), {state} (the
state of a per-state panel, trajectory_panel_<state>) and {module}; they are filled in when the figure is
drawn, so the style keeps working when the user switches problems. Whenever the user means "the system /
problem name" (not one fixed text), write the placeholder, never today's name: e.g. "system name as the
title" -> "title": {"text": "{system}"} (for every figure use "*"), "<system> trajectories" ->
"{system} trajectories". This also works in plot_style text keys.
A title needs room: with a title, also set "margin": {"t": 60} (or more for large fonts) so it is not cut off.
Do not say something cannot be styled: if Plotly can do it, do it with figure_style. If no figure is on
screen yet, use the figure ids (they apply when the figure appears).
"""

REPLY_RULES = """Answer with ONE JSON object and nothing else. "reply" (1-3 short plain-text sentences for the user:
what you set up or changed and any value you assumed) is always required; every other key is optional and
only what you include changes.
- "reply" language: the language of the user's latest request; if that is unclear (a name, a formula,
  a one-word message), the user's interface language given below.
- Write "reply" as plain text: real characters (θ, ω) or names (theta), never \\u escapes, LaTeX or markdown.
- Settings the user did not ask about stay as they are. If asked for good settings, choose them and say why briefly.
"""

FORWARD_INSTRUCTIONS = INTRO + """
You are in the Forward Solver module, which solves an autonomous initial-value problem
dx_i/dt = f_i(x; p) on [t0, t1] with adalib and compares it with a scipy reference solution. You define
that problem and its solver settings, and you style the trajectory figures (for papers and reports);
the user reviews it and presses Run.

""" + SYSTEM_RULES + """
Answer format:
{
  "reply": "...",
  "system": {
    "name": "snake_case_name",
    "states": [{"name": "x", "initial": 1.0}],
    "parameters": [{"name": "k", "value": 2.0}],
    "equations": ["-k*x"]
  },
  "time": {"t0": 0.0, "t1": 5.0},
  "options": {"n_seg": 100, "Nt_total": 5000},
  "plot_style": {"colors": {"x": "#d62728"}, "line_width": 2}
}
"system" and "time" may be omitted to keep the current ones (e.g. for a figure-only or solver-only request).
"options" and "plot_style" contain only what changes. No control inputs in this module.

- Solver settings ("options", adalib.ForwardOptions). Change them only when the user asks (e.g. "more
  segments", "use LPA", "turn off L-BFGS", "10 epochs") and keep the rest. Keys:
  basis "adaf" (ADA-F: adaptive Fourier on n_seg sequential time segments) or "lpa" (Legendre, whole span);
  n_seg segments (ADA-F, 1-2000; more for longer spans or faster dynamics); Nt_total collocation points
  over the whole span (2-200000; must be divisible by n_seg for ADA-F, so adjust it with n_seg and say so);
  epochs Adam outer epochs (0-10000); adam_inner Adam steps per epoch (1-100000); adam_lr (0-1];
  use_lbfgs true/false (L-BFGS polish after Adam); N_p Fourier/Legendre modes (1-500); N_m Fourier
  truncation (ADA-F, 1-2000); order LPA polynomial order (1-50); dtype "float64" or "float32";
  gamma (>0) and L basis time-scale parameters.
- Figure style ("plot_style"), applied at once to the Trajectories plots without re-running. Include only
  the keys to change; set a key to null to restore its default. Keys:
  font_family ("Times New Roman", "Arial", ...), font_size / title_font_size / tick_font_size /
  legend_font_size (pt); line_width (ADA lines); colors {state: colour} and dashes {state: "solid"|"dash"|
  "dot"|"dashdot"|"longdash"} for the ADA line of each state; show_reference (bool), ref_color,
  ref_line_width, ref_dash for the reference solution; title, x_title, y_title (one plot); y_titles and
  panel_titles {state: text} (per-state panels); x_range / y_range [low, high]; log_y; grid;
  legend_position "top"|"top-left"|"top-right"|"bottom-left"|"bottom-right"|"right"|"hidden"; legend_box;
  ada_label / ref_label (legend names); export_width_cm, export_height_cm, export_dpi (72-1200),
  export_format "png"|"svg" for saving. Colours: hex (#d62728), CSS names (red) or rgb(...).
  Per-state maps use the state names exactly. Put units in axis titles ("Time [s]").
  plot_style covers the common Trajectories settings (and the export size of the per-state panels); for
  anything else, or any other figure, use "figure_style" below.
""" + FIGURE_RULES + "\n" + REPLY_RULES

# kept for callers/tests that refer to the Forward prompt by its old name
INSTRUCTIONS = FORWARD_INSTRUCTIONS

CONTROL_SYSTEM = """The system here has control inputs:
  "system": {
    "name": "snake_case_name",
    "states": [{"name": "x", "initial": 1.0, "lower": -2.0, "upper": 2.0}],
    "parameters": [{"name": "k", "value": 1.0}],
    "controls": [{"name": "u", "value": 0.0, "lower": -1.0, "upper": 1.0}],
    "equations": ["-k*x + u"]
  }
- Every state needs a range lower < upper: the box the surrogate is trained on. Make it cover the states'
  expected values with some margin (e.g. 20 %), and contain the initial values.
- Controls u (at most 10) are held constant over each step; lower < upper bound their values, "value" is a
  nominal value inside the bounds. Equations may use control names. Most control problems have at least
  one control; if the user's model has none, add the natural actuator (a force, an inflow, a heat input).
"""

OPERATOR_INSTRUCTIONS = INTRO + """
You are in the Operator Learning module. It trains a neural operator (surrogate) of an ODE with control
inputs from the physics residual alone (no data): given the state x at the start of a segment and controls u
held constant over it, it predicts the state at the end of the segment, of length t_final / n_seg, and is
rolled out over n_seg segments. You set up that training; the user reviews it and presses Train.
Built-in legacy problems ("source": "builtin", "builtin": "cstr" | "triple_tank" | "fedbatch_bioreactor")
use their own equations, ranges and time grid; everything else uses "source": "custom" and "system".

""" + SYSTEM_RULES + "\n" + CONTROL_SYSTEM + """
Answer format (state values on screen are given below in the same form):
{
  "reply": "...",
  "source": "custom",
  "builtin": "triple_tank",
  "system": {...},
  "n_seg": 25,
  "t_final": 5.0,
  "options": {"epochs": 2000}
}
- n_seg (1-1000) rollout segments and t_final (> 0) the rollout horizon (custom only); a segment length
  t_final / n_seg should resolve the dynamics (a fraction of the fastest time constant).
- "options" (adalib.OperatorOptions), only what changes: n_train training samples (2-200000), n_val
  validation samples (<= n_train), epochs (1-100000), batch_size, lr learning rate, hidden layer width
  (4-4096), n_layers (1-12), seed; custom path: lpa_n_panels (1-64), lpa_max_order (1-20), lpa_nt_seg
  (4-400) for the LPA basis per segment; built-in path: basis "lpa"|"adaf", use_lr_schedule.
  More samples, epochs and width train longer but more accurately.

""" + FIGURE_RULES + "\n" + REPLY_RULES

INVERSE_INSTRUCTIONS = INTRO + """
You are in the Inverse Problem module. It estimates unknown parameters of an ODE from observations of
the states with adalib (ADA-F fit with a physics loss and a data loss). Observations are either generated
(synthetic: a scipy solution with the parameter values entered, sampled and with noise) or a CSV file the
user uploaded (you cannot create a file). You set up the problem; the user reviews it and presses Run.

""" + SYSTEM_RULES + """
Answer format (state values on screen are given below in the same form):
{
  "reply": "...",
  "system": {"name": "...", "states": [{"name": "x", "initial": 1.0}], "parameters": [{"name": "k", "value": 2.0}],
             "equations": ["-k*x"]},
  "time": {"t0": 0.0, "t1": 5.0},
  "unknowns": {"k": {"estimate": true, "initial": 1.0, "lower": 0.0, "upper": null}},
  "data_source": "synthetic",
  "synthetic": {"n_points": 100, "noise_pct": 1.0, "seed": 42, "observed": []},
  "compare_to_values": false,
  "options": {"epochs": 10}
}
- No control inputs. parameters[].value are the TRUE values that generate synthetic data, and the fixed
  values of parameters that are not estimated.
- "unknowns" maps a parameter name to what changes for it: estimate (true = unknown, to be estimated),
  initial (the starting guess, deliberately off the true value, e.g. half or double), lower / upper bounds
  (null = unbounded; lower < initial < upper). Positive physical constants get lower 0.
  At least one parameter must be estimated.
- "synthetic": n_points (2-100000) samples over the time span, noise_pct Gaussian noise in % of each
  state's RMS (0-100), seed, observed = the observed states (empty = all).
- "data_source": "synthetic" or "upload" ("upload" only when a file was uploaded, see "uploaded_file").
  "compare_to_values" (uploads): report errors against the parameter values entered.
- "options" (adalib.InverseOptions), only what changes: lambda_physics / lambda_data loss weights (>= 0);
  n_seg (1-2000), N_p (1-500), N_m (1-2000), Nt_total (2-200000), gamma (> 0), L for the ADA-F basis;
  epochs (0-10000), adam_inner (1-100000), adam_lr (0-1], use_lbfgs; n_passes, warm_seg_passes,
  n_warm_segs; training_strategy "joint"|"alternating"; data_prefit_steps; normalize_data_loss,
  normalize_physics_loss (bool); dtype "float64"|"float32".

""" + FIGURE_RULES + "\n" + REPLY_RULES

MPC_INSTRUCTIONS = INTRO + """
You are in the MPC module. It first trains a surrogate operator of an ODE with control inputs over the
state/control box (one control interval dt ahead), then runs closed-loop model-predictive control on it:
n_steps steps, each optimising the controls over a prediction horizon to track target values. Built-in
legacy problems ("source": "builtin", "builtin": "cstr" | "triple_tank" | "fedbatch_bioreactor") have their
own model; their initial values, targets and bounds are edited in the panel, not by you.
Changing the system, a range, dt or a training option needs a new surrogate (the user presses Train);
targets, weights, controllers, n_steps and horizon only need Run.

""" + SYSTEM_RULES + "\n" + CONTROL_SYSTEM + """
Answer format (state values on screen are given below in the same form):
{
  "reply": "...",
  "source": "custom",
  "builtin": "cstr",
  "system": {...},
  "dt": 0.1,
  "targets": {"x": 0.5},
  "tracking_weights": {"x": 1.0},
  "control_weights": {"u": 0.01},
  "controllers": ["autodiff"],
  "options": {"n_steps": 50, "horizon": 10}
}
- states[].initial is the closed loop's starting point. "targets" maps tracked states to their reference
  values (at least one, reachable with the control bounds); weights (>= 0) are optional, per state / control.
- dt (> 0) the control interval: a fraction of the dominant time constant. Closed-loop time = dt * n_steps.
- "controllers" (1-4, compared on the same surrogate): "autodiff" (SLSQP, exact AD gradient), "fd" (SLSQP,
  finite differences), "CEM", "MPPI" (sampling; built-in systems only).
- "options", only what changes: n_steps closed-loop steps (1-10000), horizon prediction steps (1-200, null =
  adalib default), plant_solver (scipy solve_ivp method, "RK45", "LSODA", "Radau", ...); surrogate
  training: n_train (2-200000), n_val (<= n_train), epochs, batch_size, lr, hidden (4-4096), n_layers (1-12),
  seed, lpa_n_panels, lpa_max_order, lpa_nt_seg (custom), basis "lpa"|"adaf" and use_lr_schedule (built-in).

""" + FIGURE_RULES + "\n" + REPLY_RULES


# ── messages ─────────────────────────────────────────────────────────────

def _lang(language: Optional[str]) -> str:
    return f"User interface language: {language}.\n" if language else ""


def _figures(req, current: dict) -> str:
    """What the figures look like now (before the figure style) and the style set so far."""
    shown = json.dumps(req.figures, ensure_ascii=False)
    if len(shown) > 60_000:                       # very large layouts: titles, axes and traces only
        shown = json.dumps([{"id": f.get("id"), "title": f.get("title"), "traces": f.get("traces"),
                             "axes": sorted(k for k in (f.get("layout") or {}) if k[1:5] == "axis")}
                            for f in req.figures], ensure_ascii=False)
    return ("\nFigure ids of this module: " + json.dumps(req.figure_ids)
            + "\nFigures on screen (layout before the figure style): " + (shown if req.figures else "none yet")
            + "\nCurrent figure_style: " + json.dumps(current or {}, ensure_ascii=False))


def _context(req: AssistForwardRequest) -> str:
    lang = _lang(req.language)
    options = json.dumps({"options": (req.current_options or ForwardOptionsModel()).model_dump()})
    style = (req.current_plot_style or PlotStyleModel()).model_dump(exclude_none=True)
    options += "\nCurrent figure style (unset keys use the defaults): " + json.dumps({"plot_style": style}, ensure_ascii=False)
    if req.current_system is None:
        return (lang + "There is no system on screen yet. Current solver settings: " + options
                + _figures(req, req.current_figure_style))
    cur = {"system": req.current_system.model_dump(include={"name", "states", "parameters", "equations"}),
           "time": (req.current_time or TimeSpan()).model_dump()}
    return (lang + "The system currently on screen:\n" + json.dumps(cur, ensure_ascii=False)
            + "\nCurrent solver settings: " + options + _figures(req, req.current_figure_style))


def _messages(req, context: str) -> List[dict]:
    """Provider-neutral messages: {"role", "content": str, "images": [AssistImage]} (images optional)."""
    msgs = [{"role": m.role, "content": m.content} for m in req.history]
    prompt = req.prompt.strip() or "Set up the system shown in the image."
    if req.images:
        prompt += f"\n({len(req.images)} image(s) attached.)"
    msgs.append({"role": "user", "content": f"{context}\n\nRequest: {prompt}", "images": req.images})
    return msgs


def _wire(provider: str, messages: List[dict]) -> List[dict]:
    """Neutral messages -> the provider's format (images as content blocks)."""
    out = []
    for m in messages:
        images = m.get("images") or []
        if not images:
            out.append({"role": m["role"], "content": m["content"]})
        elif provider == "anthropic":
            out.append({"role": m["role"], "content": [
                *({"type": "image", "source": {"type": "base64", "media_type": im.media_type, "data": im.data}}
                  for im in images),
                {"type": "text", "text": m["content"]}]})
        else:
            out.append({"role": m["role"], "content": [
                {"type": "text", "text": m["content"]},
                *({"type": "image_url", "image_url": {"url": f"data:{im.media_type};base64,{im.data}"}}
                  for im in images)]})
    return out


# ── providers ────────────────────────────────────────────────────────────

def _post(url: str, headers: dict, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("error", {})
            detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
        except Exception:
            detail = ""
        if exc.code in (401, 403):
            raise AssistError(f"The API key was rejected ({exc.code}). Check it in the assistant settings. {detail}", 400)
        if exc.code == 404:
            raise AssistError(f"Model not found ({detail or exc.code}). Check the model name in the assistant settings.", 400)
        if exc.code == 429:
            raise AssistError(f"Rate limit or quota exceeded (429). {detail}", 429)
        raise AssistError(f"The LLM API returned {exc.code}. {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AssistError(f"Could not reach the LLM API: {getattr(exc, 'reason', exc)}")


def _call(provider: str, model: str, key: str, messages: List[dict], instructions: str = INSTRUCTIONS) -> str:
    if provider == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        out = _post(f"{base}/v1/messages",
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                    {"model": model, "max_tokens": 4000, "system": instructions,
                     "messages": _wire(provider, messages)})
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    out = _post(f"{base}/chat/completions",
                {"authorization": f"Bearer {key}"},
                {"model": model, "messages": [{"role": "system", "content": instructions}, *_wire(provider, messages)],
                 "response_format": {"type": "json_object"}, "max_completion_tokens": 8000})
    choices = out.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


_BROKEN_ESCAPE = re.compile(r"([\x01-\x08\x0b\x0c\x0e-\x1f])([0-9a-fA-F]{2})")


def _clean_reply(reply: str) -> str:
    """Repair a mis-escaped character some models emit in JSON mode ("\\u0003b8" for "\\u03b8" = θ),
    then drop any remaining control characters."""
    reply = _BROKEN_ESCAPE.sub(lambda m: chr((ord(m.group(1)) << 8) | int(m.group(2), 16)), reply)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", reply).strip()


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)          # tolerate ```json fences or a stray sentence
    if not m:
        raise ValueError("The answer was not a JSON object.")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("The answer was not a JSON object.")
    return data


def _errors(exc: ValidationError, prefix: str = "") -> str:
    return prefix + "; ".join((f"{'.'.join(map(str, e['loc']))}: " if e["loc"] else "") + e["msg"]
                              for e in exc.errors())


def _ask(req, instructions: str, context: str, parse: Callable[[dict], tuple]):
    """Ask the configured LLM, parse, and retry with the error until the answer is valid.
    Returns (parse result, provider, model, attempts)."""
    settings = _load()
    provider = req.provider or status().provider
    if provider is None:
        raise AssistError("No LLM API key is set. Add an Anthropic or OpenAI key in the assistant settings.", 400)
    key, _ = _key(provider, settings)
    if key is None:
        raise AssistError(f"No API key for {PROVIDERS[provider]['label']}. Add it in the assistant settings.", 400)
    model = _model(provider, settings)

    messages = _messages(req, context)
    problem = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = _call(provider, model, key, messages, instructions)
        try:
            return parse(_json(text)), provider, model, attempt
        except (ValueError, json.JSONDecodeError) as exc:
            problem = str(exc)
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": f"That answer is invalid: {problem}\n"
                                            "Reply again with the corrected complete JSON object only."}]
    raise AssistError(f"The model did not produce a valid setup after {MAX_ATTEMPTS} tries: {problem}", 422)


# ── shared checks ────────────────────────────────────────────────────────

def _check_keys(data: dict, allowed) -> None:
    unknown = sorted(set(data) - set(allowed) - {"reply"})
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown}; valid: {sorted(allowed)}.")


def _system(raw, *, controls: bool) -> SystemSpec:
    """A system from the answer, checked with the equation parser."""
    if not isinstance(raw, dict):
        raise ValueError('"system" must be an object.')
    if not controls and raw.get("controls"):
        raise ValueError("This module has no control inputs; use parameters instead.")
    try:
        system = SystemSpec(**{**raw, "controls": (raw.get("controls") or []) if controls else []})
    except ValidationError as exc:
        raise ValueError(_errors(exc, "system: "))
    check = validate_spec(system)
    if not check.ok:
        raise ValueError("; ".join(e.message for e in check.errors))
    return system


def _merge_model(model: type, current: dict, patch, what: str) -> BaseModel:
    """Merge a partial "options"-style object into the current values and validate it."""
    if not isinstance(patch, dict):
        raise ValueError(f'"{what}" must be an object of settings to change.')
    unknown = sorted(set(patch) - set(model.model_fields))
    if unknown:
        raise ValueError(f"Unknown {what} key(s) {unknown}; valid: {sorted(model.model_fields)}.")
    try:
        return model(**{**current, **patch})
    except ValidationError as exc:
        raise ValueError(_errors(exc, f"{what}: "))


# ── Forward ──────────────────────────────────────────────────────────────

def _parse_forward(data: dict, req: AssistForwardRequest) -> Tuple[SystemSpec, TimeSpan, ForwardOptionsModel, dict, dict, str]:
    """JSON answer -> (system, time, options, plot_style, figure_style, reply); raises ValueError with a
    message for the model."""
    if "system" not in data and req.current_system is None:
        raise ValueError('There is no system on screen, so "system" is required.')
    system = _system(data["system"], controls=False) if "system" in data else req.current_system
    try:
        time = (TimeSpan(**(data.get("time") or {})) if "time" in data or "system" in data
                else req.current_time or TimeSpan())
    except ValidationError as exc:
        raise ValueError(_errors(exc, "time: "))
    check = validate_spec(system)
    if not check.ok:
        raise ValueError("; ".join(e.message for e in check.errors))
    options = _merge_options(req.current_options or ForwardOptionsModel(), data.get("options") or {})
    style = _merge_plot_style(req.current_plot_style, data.get("plot_style") or {}, system.state_names)
    figures = merge_figure_style(req.current_figure_style, data.get("figure_style"), req.figure_ids)
    return system, time, options, style, figures, _clean_reply(str(data.get("reply", "")))


def _merge_plot_style(current: Optional[PlotStyleModel], patch, states) -> dict:
    if not isinstance(patch, dict):
        raise ValueError('"plot_style" must be an object of style keys to change.')
    unknown = sorted(set(patch) - set(PlotStyleModel.model_fields))
    if unknown:
        raise ValueError(f"Unknown plot_style key(s) {unknown}; valid: {sorted(PlotStyleModel.model_fields)}.")
    try:
        return merge_style(current.model_dump(exclude_none=True) if current else {}, patch, states)
    except ValidationError as exc:
        raise ValueError(_errors(exc, "plot_style: "))


def _merge_options(current: ForwardOptionsModel, patch) -> ForwardOptionsModel:
    return _merge_model(ForwardOptionsModel, current.model_dump(), patch, "options")


def describe_forward(req: AssistForwardRequest) -> AssistForwardResponse:
    (system, time, options, style, figures, reply), provider, model, attempts = _ask(
        req, FORWARD_INSTRUCTIONS, _context(req), lambda data: _parse_forward(data, req))
    return AssistForwardResponse(system=system, time=time, options=options, plot_style=style, figure_style=figures,
                                 reply=reply, provider=provider, model=model, attempts=attempts)


# ── Operator ─────────────────────────────────────────────────────────────

OPERATOR_KEYS = ("source", "builtin", "system", "n_seg", "t_final", "options")


def _parse_operator(data: dict, cur: dict) -> dict:
    _check_keys(data, OPERATOR_KEYS)
    new = {k: cur.get(k) for k in OPERATOR_KEYS}
    for k in ("source", "builtin", "n_seg", "t_final"):
        if k in data:
            new[k] = data[k]
    if "system" in data:
        new["system"] = _system(data["system"], controls=True).model_dump()
    new["options"] = _merge_model(OperatorOptionsModel, cur.get("options") or {}, data.get("options") or {},
                                  "options").model_dump()
    try:
        req = OperatorTrainRequest(**{k: v for k, v in new.items() if v is not None})
    except ValidationError as exc:
        raise ValueError(_errors(exc))
    if req.source == "custom":
        _system(new["system"], controls=True)
    return {**new, **req.model_dump(include={"source", "n_seg", "t_final"})}


# ── Inverse ──────────────────────────────────────────────────────────────

INVERSE_KEYS = ("system", "time", "unknowns", "data_source", "synthetic", "compare_to_values", "options")


def _parse_inverse(data: dict, cur: dict) -> dict:
    _check_keys(data, INVERSE_KEYS)
    system = _system(data["system"] if "system" in data else cur.get("system"), controls=False)
    try:
        time = TimeSpan(**(data["time"] if "time" in data else cur.get("time") or {}))
    except (ValidationError, TypeError) as exc:
        raise ValueError(_errors(exc, "time: ") if isinstance(exc, ValidationError) else '"time" must be an object.')

    # unknowns: per parameter, merged with what is on screen; parameters of a replaced system start fixed
    patch = data.get("unknowns") or {}
    if not isinstance(patch, dict):
        raise ValueError('"unknowns" must map parameter names to {estimate, initial, lower, upper}.')
    names = system.parameter_names
    stray = sorted(set(patch) - set(names))
    if stray:
        raise ValueError(f"unknowns {stray} are not parameters of the system ({names}).")
    old = cur.get("unknowns") or {}
    unknowns = {}
    for p in system.parameters:
        u = {"estimate": False, "initial": p.value, "lower": None, "upper": None, **(old.get(p.name) or {})}
        if not isinstance(patch.get(p.name, {}), dict):
            raise ValueError(f'unknowns.{p.name} must be an object {{estimate, initial, lower, upper}}.')
        u.update(patch.get(p.name) or {})
        if u["estimate"]:
            try:
                UnknownParameter(name=p.name, initial=u["initial"], lower=u["lower"], upper=u["upper"])
            except ValidationError as exc:
                raise ValueError(_errors(exc, f"unknowns.{p.name}: "))
        unknowns[p.name] = {k: u[k] for k in ("estimate", "initial", "lower", "upper")}
    if not any(u["estimate"] for u in unknowns.values()):
        raise ValueError("Mark at least one parameter to estimate (unknowns.<name>.estimate = true).")

    syn_cur = {"n_points": 100, "noise_pct": 1.0, "seed": 42, "observed": [], **(cur.get("synthetic") or {})}
    syn_cur["observed"] = [s for s in syn_cur.get("observed") or [] if s in system.state_names]
    synthetic = _merge_model(SyntheticObservations, syn_cur, data.get("synthetic") or {}, "synthetic")
    stray = sorted(set(synthetic.observed) - set(system.state_names))
    if stray:
        raise ValueError(f"synthetic.observed {stray} are not states ({system.state_names}).")
    source = data.get("data_source", cur.get("data_source") or "synthetic")
    if source not in ("synthetic", "upload"):
        raise ValueError('"data_source" must be "synthetic" or "upload".')
    if source == "upload" and not cur.get("uploaded_file"):
        raise ValueError('No file is uploaded, so "data_source" must be "synthetic".')
    options = _merge_model(InverseOptionsModel, cur.get("options") or {}, data.get("options") or {}, "options")
    return {"system": system.model_dump(), "time": time.model_dump(), "unknowns": unknowns,
            "data_source": source,
            "synthetic": synthetic.model_dump(include={"n_points", "noise_pct", "seed", "observed"}),
            "compare_to_values": bool(data.get("compare_to_values", cur.get("compare_to_values", False))),
            "options": options.model_dump()}


# ── MPC ──────────────────────────────────────────────────────────────────

MPC_KEYS = ("source", "builtin", "system", "dt", "targets", "tracking_weights", "control_weights",
            "controllers", "options")


def _weights(raw, names: List[str], what: str) -> Dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f'"{what}" must map names to numbers.')
    out = {}
    for n, v in raw.items():
        if n not in names:
            raise ValueError(f"{what}: '{n}' is not one of {names}.")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
            raise ValueError(f"{what}.{n} must be a number.")
        if what != "targets" and v < 0:
            raise ValueError(f"{what}.{n} must be >= 0.")
        out[n] = float(v)
    return out


def _parse_mpc(data: dict, cur: dict) -> dict:
    _check_keys(data, MPC_KEYS)
    new = {k: cur.get(k) for k in MPC_KEYS}
    for k in ("source", "builtin", "dt", "controllers"):
        if k in data:
            new[k] = data[k]
    if "system" in data:
        new["system"] = _system(data["system"], controls=True).model_dump()
    options = _merge_model(MPCOptionsModel, cur.get("options") or {}, data.get("options") or {}, "options")
    new["options"] = options.model_dump()
    if new["source"] not in ("custom", "builtin"):
        raise ValueError('"source" must be "custom" or "builtin".')
    if new["builtin"] not in BUILTINS:
        raise ValueError(f'"builtin" must be one of {list(BUILTINS)}.')
    ctrl = new["controllers"]
    if not isinstance(ctrl, list) or not ctrl or any(c not in MPC_CONTROLLERS for c in ctrl) or len(set(ctrl)) != len(ctrl):
        raise ValueError(f'"controllers" must list 1-4 different ids of {list(MPC_CONTROLLERS)}.')
    try:
        train = MPCTrainRequest(source=new["source"], builtin=new["builtin"], dt=new["dt"],
                                system=new["system"] if new["source"] == "custom" else None,
                                options={k: v for k, v in new["options"].items() if k in MPCSurrogateOptions.model_fields})
    except ValidationError as exc:
        raise ValueError(_errors(exc))
    new["dt"] = train.dt
    if new["source"] == "builtin":
        return new                      # built-in targets/bounds are edited in the panel
    system = _system(new["system"], controls=True)
    states, controls = system.state_names, system.control_names
    # names a replaced system no longer has are dropped
    for k, names in (("targets", states), ("tracking_weights", states), ("control_weights", controls)):
        given = data.get(k) if k in data else {n: v for n, v in (cur.get(k) or {}).items() if n in names}
        new[k] = _weights(given or {}, names, k)
    if not new["targets"]:
        raise ValueError('Choose at least one state to track: "targets": {state: reference value}.')
    if any(c in ("CEM", "MPPI") for c in ctrl):
        raise ValueError('Custom systems use "fd" or "autodiff" controllers (CEM/MPPI are built-in only).')
    return new


# ── generic entry point ──────────────────────────────────────────────────

MODULES = {
    "operator": (OPERATOR_INSTRUCTIONS, _parse_operator, "Operator Learning setup"),
    "inverse": (INVERSE_INSTRUCTIONS, _parse_inverse, "Inverse Problem setup"),
    "mpc": (MPC_INSTRUCTIONS, _parse_mpc, "MPC setup"),
}


def describe_module(module: str, req: AssistModuleRequest) -> AssistModuleResponse:
    instructions, parse, title = MODULES[module]
    current = {k: v for k, v in req.state.items() if k != "figure_style"}
    figure_style = req.state.get("figure_style") or {}
    context = (_lang(req.language) + f"The {title} currently on screen:\n"
               + json.dumps(current, ensure_ascii=False) + _figures(req, figure_style))

    def _do(data: dict):
        figures = merge_figure_style(figure_style, data.pop("figure_style", None), req.figure_ids)
        return {**parse(data, current), "figure_style": figures}, _clean_reply(str(data.get("reply", "")))
    (state, reply), provider, model, attempts = _ask(req, instructions, context, _do)
    return AssistModuleResponse(state=state, reply=reply, provider=provider, model=model, attempts=attempts)
