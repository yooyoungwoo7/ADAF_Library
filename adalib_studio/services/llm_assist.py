"""
Natural-language system definition for the Forward module.

The user's sentence (plus what is on screen and the chat so far) goes to an
LLM (Anthropic Claude or OpenAI) that answers with a JSON system definition.
The answer is checked with the same equation parser the editor uses; if it
fails, the error is sent back once or twice for a corrected answer. The result
only fills the editor: the user still reviews it and presses Run.

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
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from ..config import SETTINGS_FILE
from ..schemas.assist import (AssistForwardRequest, AssistForwardResponse, AssistProviderStatus,
                              AssistSettingsUpdate, AssistStatus)
from ..schemas.common import SystemSpec, TimeSpan
from .equation_parser import CONSTANTS, FUNCTIONS
from .system_builder import validate_spec

PROVIDERS: Dict[str, Dict[str, str]] = {
    "anthropic": {"label": "Anthropic (Claude)", "env": "ANTHROPIC_API_KEY", "default_model": "claude-haiku-5-5"},
    "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "default_model": "gpt-5.4-mini"},
}
MAX_ATTEMPTS = 3
TIMEOUT_S = 90


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


# ── prompt ───────────────────────────────────────────────────────────────

INSTRUCTIONS = f"""You are the assistant inside ADALib Studio, the browser workspace of adalib (PyPI package
"adalib-IFL", the Anti-Derivative Approximator library). adalib solves ODE systems in a physics-informed way:
it fits an ADA basis (ADA-F: adaptive Fourier on sequential time segments, or LPA: Legendre polynomials)
so that the ODE residual is minimised, and Studio compares the result with a scipy reference solution.
The users are engineers and researchers who already know they are using adalib; do not explain it.

You are in the Forward Solver module, which solves an autonomous initial-value problem
dx_i/dt = f_i(x; p) on [t0, t1]. Your job is only to define that problem; the user reviews it and presses Run.

Users often just name a system ("spring-damper", "Lotka-Volterra", "Duffing", "SIR model"), possibly in
Korean. Then infer the standard model and choose, without asking questions:
- the usual state variables and parameter names (e.g. x, v and m, c, k for a mass-spring-damper),
- textbook parameter values and a typical non-trivial initial condition,
- a time span that shows the characteristic behaviour (a few oscillation periods, the settling time,
  one epidemic wave), not much longer,
and mention these choices briefly in "reply".

Answer with ONE JSON object and nothing else:
{{
  "reply": "1-3 short plain-text sentences for the user: what you set up and any value you assumed",
  "system": {{
    "name": "snake_case_name",
    "states": [{{"name": "x", "initial": 1.0}}],
    "parameters": [{{"name": "k", "value": 2.0}}],
    "equations": ["-k*x"]
  }},
  "time": {{"t0": 0.0, "t1": 5.0}}
}}

Rules:
- equations[i] is the right-hand side of d(states[i])/dt, same order and count as states.
- Rewrite higher-order ODEs as first-order systems (e.g. theta'' -> states theta, omega with theta' = omega).
- Python expression syntax: + - * / and ** for powers (never ^), numbers, state and parameter names,
  functions {", ".join(sorted(FUNCTIONS))}, constants {", ".join(sorted(CONSTANTS))}.
- The time variable t must NOT appear (only autonomous systems); forcing that depends on t is not possible.
- Names: ASCII letters, digits and _, not starting with a digit or _, unique, and none of:
  {", ".join(sorted(FUNCTIONS))}, {", ".join(sorted(CONSTANTS))}, t.
- Put every physical constant in "parameters" (do not hard-code it in the equations) so the user can edit it.
- Values the user did not give: choose standard textbook values and say so in "reply".
- At most 20 states and 50 parameters; t1 > t0. No control inputs.
- If an image is attached, it shows the model (typeset or handwritten equations, a screenshot of a paper):
  transcribe it faithfully. Keep its symbols as names where valid (Greek letters spelled out: alpha, omega),
  take parameter values and initial conditions from it, and choose only what it leaves out (say so).
  Text with the image refines it ("only the first two equations", "use k = 3").
- If the user names or describes a different system, replace the one on screen.
- If the user asks to modify the current system (a value, a term, the time span), change only that and
  return the complete updated system.
  If the message asks for no change (a greeting, a question), return the current system unchanged.
- "reply" language: the language of the user's latest request; if that is unclear (a name, a formula,
  a one-word message), the user's interface language given below.
- Write "reply" as plain text: real characters (θ, ω) or names (theta), never \\u escapes, LaTeX or markdown.
"""


def _context(req: AssistForwardRequest) -> str:
    lang = f"User interface language: {req.language}.\n" if req.language else ""
    if req.current_system is None:
        return lang + "There is no system on screen yet."
    cur = {"system": req.current_system.model_dump(include={"name", "states", "parameters", "equations"}),
           "time": (req.current_time or TimeSpan()).model_dump()}
    return lang + "The system currently on screen:\n" + json.dumps(cur, ensure_ascii=False)


def _messages(req: AssistForwardRequest) -> List[dict]:
    """Provider-neutral messages: {"role", "content": str, "images": [AssistImage]} (images optional)."""
    msgs = [{"role": m.role, "content": m.content} for m in req.history]
    prompt = req.prompt.strip() or "Set up the system shown in the image."
    if req.images:
        prompt += f"\n({len(req.images)} image(s) attached.)"
    msgs.append({"role": "user", "content": f"{_context(req)}\n\nRequest: {prompt}", "images": req.images})
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


def _call(provider: str, model: str, key: str, messages: List[dict]) -> str:
    if provider == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        out = _post(f"{base}/v1/messages",
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                    {"model": model, "max_tokens": 4000, "system": INSTRUCTIONS,
                     "messages": _wire(provider, messages)})
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    out = _post(f"{base}/chat/completions",
                {"authorization": f"Bearer {key}"},
                {"model": model, "messages": [{"role": "system", "content": INSTRUCTIONS}, *_wire(provider, messages)],
                 "response_format": {"type": "json_object"}, "max_completion_tokens": 8000})
    choices = out.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


_BROKEN_ESCAPE = re.compile(r"([\x01-\x08\x0b\x0c\x0e-\x1f])([0-9a-fA-F]{2})")


def _clean_reply(reply: str) -> str:
    """Repair a mis-escaped character some models emit in JSON mode ("\\u0003b8" for "\\u03b8" = θ),
    then drop any remaining control characters."""
    reply = _BROKEN_ESCAPE.sub(lambda m: chr((ord(m.group(1)) << 8) | int(m.group(2), 16)), reply)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", reply).strip()


def _parse(text: str) -> Tuple[SystemSpec, TimeSpan, str]:
    """JSON answer -> (system, time, reply); raises ValueError with a message for the model."""
    m = re.search(r"\{.*\}", text, re.S)          # tolerate ```json fences or a stray sentence
    if not m:
        raise ValueError("The answer was not a JSON object.")
    data = json.loads(m.group(0))
    try:
        system = SystemSpec(**{**data["system"], "controls": []})
        time = TimeSpan(**data.get("time", {}))
    except KeyError as exc:
        raise ValueError(f"Missing field {exc}.")
    except ValidationError as exc:
        raise ValueError("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
    check = validate_spec(system)
    if not check.ok:
        raise ValueError("; ".join(e.message for e in check.errors))
    return system, time, _clean_reply(str(data.get("reply", "")))


def describe_forward(req: AssistForwardRequest) -> AssistForwardResponse:
    settings = _load()
    provider = req.provider or status().provider
    if provider is None:
        raise AssistError("No LLM API key is set. Add an Anthropic or OpenAI key in the assistant settings.", 400)
    key, _ = _key(provider, settings)
    if key is None:
        raise AssistError(f"No API key for {PROVIDERS[provider]['label']}. Add it in the assistant settings.", 400)
    model = _model(provider, settings)

    messages = _messages(req)
    problem = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = _call(provider, model, key, messages)
        try:
            system, time, reply = _parse(text)
            return AssistForwardResponse(system=system, time=time, reply=reply, provider=provider,
                                         model=model, attempts=attempt)
        except (ValueError, json.JSONDecodeError) as exc:
            problem = str(exc)
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": f"That answer is invalid: {problem}\n"
                                            "Reply again with the corrected complete JSON object only."}]
    raise AssistError(f"The model did not produce a valid system after {MAX_ATTEMPTS} tries: {problem}", 422)
