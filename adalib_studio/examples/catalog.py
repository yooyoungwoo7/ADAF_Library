"""
Built-in example systems for the equation editor.

Every value is traceable to adalib-IFL 0.1.0 (installed package or the
ADAF_Library repository); `source` says where. Values chosen by ADALib Studio
itself are listed in `studio_choices`.
"""
from __future__ import annotations

from typing import Dict, List

EXAMPLES: List[Dict] = [
    {
        "id": "lotka_volterra_paper",
        "title": "Lotka–Volterra (paper scaling)",
        "modules": ["forward"],
        "description": "Predator–prey model in the scaled form used in the ADALib paper: "
                       "U = prey/200, R = predator/200, with alpha = 2R, beta = 0.04RU, "
                       "gamma = 1.06R, delta = 0.02RU for U = 200, R = 20.",
        "source": "ADAF_Library examples/forward/lotka_volterra_forward.py; "
                  "adalib.systems.lotka_volterra.LotkaVolterra (same equations)",
        "studio_choices": [],
        "system": {
            "name": "lotka_volterra",
            "states": [{"name": "U", "initial": 0.5}, {"name": "R", "initial": 0.075}],
            "parameters": [{"name": "alpha", "value": 40.0}, {"name": "beta", "value": 160.0},
                           {"name": "gamma", "value": 21.2}, {"name": "delta", "value": 80.0}],
            "equations": ["alpha*U - beta*U*R", "-gamma*R + delta*U*R"],
        },
        "time": {"t0": 0.0, "t1": 1.0},
        "options": {"basis": "adaf", "n_seg": 50, "N_p": 5, "N_m": 100, "Nt_total": 2500,
                    "epochs": 5, "adam_inner": 100, "use_lbfgs": True, "dtype": "float64"},
        "phase": ["U", "R"],
    },
    {
        "id": "euler_rigid_body",
        "title": "Euler rigid body",
        "modules": ["forward"],
        "description": "Torque-free rotation of a rigid body (Euler's equations) with principal "
                       "moments of inertia I1, I2, I3.",
        "source": "ADAF_Library examples/simple_api/02_forward_euler_example.py; "
                  "adalib.systems.euler.EulerRigidBody (same equations and defaults)",
        "studio_choices": [],
        "system": {
            "name": "euler_rigid_body",
            "states": [{"name": "w1", "initial": 1.0}, {"name": "w2", "initial": 1.0},
                       {"name": "w3", "initial": 1.0}],
            "parameters": [{"name": "I1", "value": 0.2}, {"name": "I2", "value": 0.3},
                           {"name": "I3", "value": 0.4}],
            "equations": ["(I2 - I3)/(I2*I3)*w2*w3", "(I3 - I1)/(I1*I3)*w1*w3",
                          "(I1 - I2)/(I1*I2)*w1*w2"],
        },
        "time": {"t0": 0.0, "t1": 2.5},
        "options": {"basis": "adaf", "n_seg": 20, "N_p": 10, "N_m": 100, "Nt_total": 500,
                    "epochs": 5, "adam_inner": 100, "use_lbfgs": True, "dtype": "float64"},
        "phase": ["w1", "w2"],
    },
    {
        "id": "damped_pendulum",
        "title": "Damped pendulum",
        "modules": ["forward"],
        "description": "Nonlinear pendulum with viscous damping: theta' = omega, "
                       "omega' = -gamma*omega - g_over_l*sin(theta).",
        "source": "adalib.systems.pendulum.DampedPendulum (equations and parameter defaults)",
        "studio_choices": ["initial conditions", "time interval", "solver options"],
        "system": {
            "name": "damped_pendulum",
            "states": [{"name": "theta", "initial": 1.0}, {"name": "omega", "initial": 0.0}],
            "parameters": [{"name": "gamma", "value": 0.3}, {"name": "g_over_l", "value": 9.81}],
            "equations": ["omega", "-gamma*omega - g_over_l*sin(theta)"],
        },
        "time": {"t0": 0.0, "t1": 5.0},
        "options": {"basis": "adaf", "n_seg": 50, "N_p": 5, "N_m": 100, "Nt_total": 2500,
                    "epochs": 5, "adam_inner": 100, "use_lbfgs": True, "dtype": "float64"},
        "phase": ["theta", "omega"],
    },
]


# Operator learning, custom path: the generic operator learns (x0, u) -> trajectory
# over a training box, so examples carry state ranges and control inputs.
EXAMPLES += [
    {
        "id": "pendulum_torque_operator",
        "title": "Damped pendulum with torque",
        "modules": ["operator"],
        "description": "Damped pendulum driven by a constant torque tau. The operator learns trajectories "
                       "for any start (theta, omega) and torque inside the training box.",
        "source": "equations and gamma, g_over_l from adalib.systems.pendulum.DampedPendulum",
        "studio_choices": ["torque input", "training ranges", "horizon", "initial condition"],
        "system": {
            "name": "pendulum_torque",
            "states": [{"name": "theta", "initial": 1.0, "lower": -2.0, "upper": 2.0},
                       {"name": "omega", "initial": 0.0, "lower": -4.0, "upper": 4.0}],
            "parameters": [{"name": "gamma", "value": 0.3}, {"name": "g_over_l", "value": 9.81}],
            "controls": [{"name": "tau", "value": 0.0, "lower": -2.0, "upper": 2.0}],
            "equations": ["omega", "-gamma*omega - g_over_l*sin(theta) + tau"],
        },
        "time": {"t0": 0.0, "t1": 5.0},
        "operator": {"n_seg": 25, "t_final": 5.0},
        "options": {},
    },
    {
        "id": "msd_force_operator",
        "title": "Mass-spring-damper with force",
        "modules": ["operator"],
        "description": "Linear oscillator m x'' + c x' + k x = F with an external force input F.",
        "source": "textbook system; all values chosen by ADALib Studio",
        "studio_choices": ["all values"],
        "system": {
            "name": "mass_spring_damper",
            "states": [{"name": "x", "initial": 1.0, "lower": -2.0, "upper": 2.0},
                       {"name": "v", "initial": 0.0, "lower": -3.0, "upper": 3.0}],
            "parameters": [{"name": "k", "value": 4.0}, {"name": "c", "value": 0.4}, {"name": "m", "value": 1.0}],
            "controls": [{"name": "F", "value": 0.0, "lower": -2.0, "upper": 2.0}],
            "equations": ["v", "(F - c*v - k*x)/m"],
        },
        "time": {"t0": 0.0, "t1": 5.0},
        "operator": {"n_seg": 25, "t_final": 5.0},
        "options": {},
    },
]

# Inverse problems: values and options from adalib's own examples/inverse/*.py.
EXAMPLES += [
    {
        "id": "lotka_volterra_inverse",
        "title": "Lotka-Volterra: estimate alpha, gamma",
        "modules": ["inverse"],
        "description": "Recover the prey birth rate alpha and predator death rate gamma from 200 noisy "
                       "observations of both populations; beta and delta are known.",
        "source": "ADAF_Library examples/inverse/lotka_volterra_inverse.py (values, options, 1 % noise, seed 42)",
        "studio_choices": [],
        "system": {
            "name": "lotka_volterra",
            "states": [{"name": "prey", "initial": 0.5}, {"name": "predator", "initial": 0.075}],
            "parameters": [{"name": "alpha", "value": 2.0}, {"name": "beta", "value": 0.04},
                           {"name": "gamma", "value": 1.06}, {"name": "delta", "value": 0.02}],
            "equations": ["alpha*prey - beta*prey*predator", "delta*prey*predator - gamma*predator"],
        },
        "time": {"t0": 0.0, "t1": 1.0},
        "inverse": {
            "unknowns": [{"name": "alpha", "initial": 1.5, "lower": 0.0, "upper": None},
                         {"name": "gamma", "initial": 1.2, "lower": 0.0, "upper": None}],
            "synthetic": {"n_points": 200, "noise_pct": 1.0, "seed": 42},
            "options": {"n_seg": 10, "N_p": 5, "N_m": 100, "Nt_total": 1000, "gamma": 0.8, "L": 1.0,
                        "lambda_physics": 1.0, "lambda_data": 10.0, "epochs": 5, "adam_inner": 200,
                        "adam_lr": 1e-3, "use_lbfgs": True, "dtype": "float64"},
        },
        "options": {},
    },
    {
        "id": "euler_inverse",
        "title": "Euler rigid body: estimate I2, I3",
        "modules": ["inverse"],
        "description": "Recover two principal moments of inertia from noisy angular velocities; I1 is "
                       "fixed as the identifiability anchor.",
        "source": "ADAF_Library examples/inverse/euler_inverse.py (values and bounds); observations here use "
                  "the Studio's Radau generator instead of the example's ADA forward run",
        "studio_choices": ["observation generator", "noise level"],
        "system": {
            "name": "euler_rigid_body",
            "states": [{"name": "w1", "initial": 1.0}, {"name": "w2", "initial": 1.0}, {"name": "w3", "initial": 1.0}],
            "parameters": [{"name": "I1", "value": 0.2}, {"name": "I2", "value": 0.3}, {"name": "I3", "value": 0.4}],
            "equations": ["(I2 - I3)/(I2*I3)*w2*w3", "(I3 - I1)/(I1*I3)*w1*w3", "(I1 - I2)/(I1*I2)*w1*w2"],
        },
        "time": {"t0": 0.0, "t1": 2.5},
        "inverse": {
            "unknowns": [{"name": "I2", "initial": 0.25, "lower": 0.05, "upper": 2.0},
                         {"name": "I3", "initial": 0.35, "lower": 0.05, "upper": 2.0}],
            "synthetic": {"n_points": 200, "noise_pct": 1.0, "seed": 0},
            "options": {"n_seg": 10, "N_p": 5, "N_m": 100, "Nt_total": 1000, "epochs": 5, "adam_inner": 200},
        },
        "options": {},
    },
]

# MPC, custom path: tracking with the generic surrogate.
EXAMPLES += [
    {
        "id": "pendulum_torque_mpc",
        "title": "Pendulum: hold theta at 0.2",
        "modules": ["mpc"],
        "description": "Bring a damped pendulum from rest to theta = 0.2 rad and hold it with a bounded "
                       "torque. Holding needs tau = g_over_l sin(0.2), about 1.95.",
        "source": "equations and gamma, g_over_l from adalib.systems.pendulum.DampedPendulum",
        "studio_choices": ["torque input and bounds", "target", "training box", "weights", "horizon"],
        "system": {
            "name": "pendulum_torque",
            "states": [{"name": "theta", "initial": 0.0, "lower": -1.0, "upper": 1.0},
                       {"name": "omega", "initial": 0.0, "lower": -3.0, "upper": 3.0}],
            "parameters": [{"name": "gamma", "value": 0.3}, {"name": "g_over_l", "value": 9.81}],
            "controls": [{"name": "tau", "value": 0.0, "lower": -3.0, "upper": 3.0}],
            "equations": ["omega", "-gamma*omega - g_over_l*sin(theta) + tau"],
        },
        "time": {"t0": 0.0, "t1": 3.0},
        "mpc": {"dt": 0.1, "targets": {"theta": 0.2}, "tracking_weights": {"theta": 1.0},
                "control_weights": {"tau": 0.001},
                "options": {"n_steps": 30, "horizon": 5, "gradient": "autodiff", "n_train": 400, "n_val": 80,
                            "epochs": 200, "hidden": 128, "n_layers": 3}},
        "options": {},
    },
]

# Built-in MPC problems (legacy cstr_mpc_op). x0/targets from adalib's
# examples/mpc/surrogate_mpc_showcase.py where available.
MPC_BUILTINS: List[Dict] = [
    {
        "id": "cstr", "title": "CSTR — reactor temperature tracking", "mode": "tracking",
        "description": "Track the reactor temperature T_R with the jacket heat duty Q_dot; 1.2 min per step.",
        "states": ["C_A", "C_B", "T_R", "T_K"], "controls": ["Q_dot"], "time_unit": "min",
        "x0": [0.8, 0.5, 141.0, 141.0], "targets": {"T_R": 136.0}, "trackable": ["T_R"],
        "control_bounds": {"Q_dot": [-8500.0, 0.0]},
        "source": "x0/target: examples/mpc/surrogate_mpc_showcase.py; bounds: legacy cstr_mpc defaults",
    },
    {
        "id": "triple_tank", "title": "Triple tank — level regulation", "mode": "tracking",
        "description": "Regulate tank 3's level h3 with pump flows Q1, Q2; 12 s per step.",
        "states": ["h1", "h2", "h3"], "controls": ["Q1", "Q2"], "time_unit": "s",
        "x0": [190.0, 100.0, 140.0], "targets": {"h3": 150.0}, "trackable": ["h1", "h2", "h3"],
        "control_bounds": {"Q1": [0.0, 100.0], "Q2": [0.0, 100.0]},
        "source": "x0/target: examples/mpc/surrogate_mpc_showcase.py; bounds: legacy triple_tank_mpc defaults",
    },
    {
        "id": "fedbatch_bioreactor", "title": "Fed-batch bioreactor — economic", "mode": "economic",
        "description": "Choose the feed rate inp to maximise the final product amount Ps x Vs (economic MPC, "
                       "shrinking horizon); 2 min per step. The objective is fixed by adalib.",
        "states": ["Xs", "Ss", "Ps", "Vs"], "controls": ["inp"], "time_unit": "min",
        "x0": [1.0, 0.5, 0.0, 1.0], "targets": {}, "trackable": [],
        "control_bounds": {"inp": [0.005, 0.2]},
        "source": "bounds: legacy bioreactor defaults; x0 chosen by ADALib Studio inside the training ranges",
    },
]

MPC_PRESETS: List[Dict] = [
    {"id": "quick", "title": "Quick look", "note": "seconds; surrogate barely trained",
     "options": {"n_train": 20, "n_val": 5, "epochs": 20, "hidden": 32, "n_layers": 2}},
    {"id": "standard", "title": "Standard", "note": "under a minute for most systems",
     "options": {"n_train": 200, "n_val": 40, "epochs": 100, "hidden": 128, "n_layers": 3}},
    {"id": "adalib", "title": "adalib defaults", "note": "minutes: 200 cases, 300 epochs, 256 x 4 network",
     "options": {"n_train": 200, "n_val": 50, "epochs": 300, "hidden": 256, "n_layers": 4}},
]

# Built-in operator problems (adalib legacy cstr_mpc_op). Ranges are copied from
# the problem classes for display before training; a trained model's manifest
# re-reads them from the installed package.
OPERATOR_BUILTINS: List[Dict] = [
    {
        "id": "cstr",
        "title": "CSTR (Klatt-Engell)",
        "description": "Van de Vusse reactor with cooling jacket: C_A, C_B, T_R, T_K. theta = [alpha, beta, F, Q] "
                       "is held constant over 25 segments of 0.02 h (0.5 h). Training initial states stay "
                       "within 5 % of the range around the nominal state.",
        "source": "adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/cstr_problem.py, config.py",
        "states": ["C_A", "C_B", "T_R", "T_K"],
        "inputs": ["alpha", "beta", "F", "Q"],
        "input_ranges": [[0.9, 1.1], [0.9, 1.1], [5.0, 100.0], [-8500.0, 0.0]],
        "time": "25 x 0.02 h (shown in min)",
    },
    {
        "id": "triple_tank",
        "title": "Triple tank",
        "description": "Three Torricelli tanks h1, h2, h3 fed by pumps Q1, Q2; 10 segments of 30 s (300 s).",
        "source": "adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/triple_tank_problem.py",
        "states": ["h1", "h2", "h3"],
        "inputs": ["Q1", "Q2"],
        "input_ranges": [[0.0, 150.0], [0.0, 150.0]],
        "time": "10 x 30 s",
    },
    {
        "id": "fedbatch_bioreactor",
        "title": "Fed-batch bioreactor",
        "description": "Biomass, substrate, product and volume (Xs, Ss, Ps, Vs); theta = [Yx, Sin, inp]; "
                       "50 segments of 2 min (100 min).",
        "source": "adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/bioreactor_problem.py",
        "states": ["Xs", "Ss", "Ps", "Vs"],
        "inputs": ["Yx", "Sin", "inp"],
        "input_ranges": [[0.3, 0.5], [0.3, 1.5], [0.005, 0.2]],
        "time": "50 x 2 min",
    },
]

# Training-size presets. "adalib defaults" = OperatorOptions() of 0.1.0.
OPERATOR_PRESETS: List[Dict] = [
    {"id": "quick", "title": "Quick look", "note": "seconds; for checking a setup, not for accuracy",
     "options": {"n_train": 50, "n_val": 10, "epochs": 50, "hidden": 64, "n_layers": 2}},
    {"id": "standard", "title": "Standard", "note": "built-ins 5-16 s (1-9 % error for CSTR / triple tank); custom ~3 s but coarse (20-30 %)",
     "options": {"n_train": 200, "n_val": 40, "epochs": 200, "hidden": 128, "n_layers": 3}},
    {"id": "adalib", "title": "adalib defaults", "note": "custom ~20 s with ~1-2 % error in tests; built-ins take tens of minutes or more",
     "options": {"n_train": 2000, "n_val": 200, "epochs": 1000, "hidden": 128, "n_layers": 3}},
]


def list_examples(module: str | None = None) -> List[Dict]:
    if module is None:
        return EXAMPLES
    return [e for e in EXAMPLES if module in e["modules"]]
