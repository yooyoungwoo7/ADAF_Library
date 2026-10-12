"""What this Studio build can actually run, per module (see docs/api_integration.md)."""
from __future__ import annotations

from .equation_parser import CONSTANTS, FUNCTIONS
from .package_info import adalib_version, forward_option_defaults


def capabilities() -> dict:
    try:
        fwd_defaults = forward_option_defaults()
    except Exception as exc:  # adalib missing: report instead of crashing the UI
        fwd_defaults = {"error": str(exc)}
    return {
        "adalib_version": adalib_version(),
        "parser": {"functions": sorted(FUNCTIONS), "constants": sorted(CONSTANTS),
                   "operators": ["+", "-", "*", "/", "^ (or **)"],
                   "time_dependence": False,
                   "time_dependence_note": "adalib 0.1.0's rhs_tf residual does not receive t, "
                                           "so only autonomous systems are accepted."},
        "modules": {
            "forward": {
                "status": "available",
                "api": "adalib.run_forward(system, x0, t_span, params, options)",
                "custom_systems": "full (CallableODESystem with generated TF-native rhs_tf; L-BFGS enabled)",
                "option_defaults": fwd_defaults,
                "progress": "per-segment (ADA-F) or per-epoch (LPA) losses parsed from adalib's verbose log",
            },
            "operator": {
                "status": "available",
                "api": "adalib.run_operator(system, x0, t_span, params, options)",
                "custom_systems": "generic physics-residual-only LPA operator; parameters fixed at training, "
                                  "control inputs held constant per rollout",
                "builtin_systems": ["cstr", "triple_tank", "fedbatch_bioreactor"],
                "progress": "epoch losses parsed from adalib's log (built-in: every 10th epoch)",
            },
            "inverse": {
                "status": "available",
                "api": "adalib.run_inverse(system, x0, t_span, params, data, options)",
                "custom_systems": "full (needs the TF-native rhs_tf the parser generates)",
                "methods": ["joint", "alternating"],
                "progress": "per-segment loss and current parameter values parsed from adalib's log",
            },
            "mpc": {
                "status": "available",
                "api": "adalib.run_mpc(system, x0, t_span, options)",
                "custom_systems": "tracking MPC (generic LPA surrogate, SLSQP); objective fixed by adalib",
                "builtin_systems": {"cstr": "tracking", "triple_tank": "tracking", "fedbatch_bioreactor": "economic"},
                "progress": "surrogate training epochs, then one line per closed-loop step",
            },
        },
    }
