"""
adalib/operator/rollout.py
Thin wrappers around OperatorLearner for forward prediction.

These expose:
    predict_step(learner, x, theta) -> x_end
    predict_rollout(learner, x0, theta_seq) -> states (horizon+1, n_state)
"""
from __future__ import annotations
import numpy as np


def predict_step(learner, x, theta):
    """
    Single-segment prediction.

    Parameters
    ----------
    learner : OperatorLearner
    x       : array (n_state,)  — current state
    theta   : array (param_dim,) — [Yx, Sin, inp] for bioreactor etc.

    Returns
    -------
    x_end : array (n_state,)
    """
    z = np.concatenate([x, theta], axis=0).astype(np.float32)[None, :]
    out = learner.predict_segment(z)
    return out["x_end"][0].astype(np.float32)


def predict_rollout(learner, x0, theta_seq):
    """
    Multi-step rollout prediction.

    Parameters
    ----------
    learner    : OperatorLearner
    x0         : array (n_state,)
    theta_seq  : array (horizon, param_dim) — params at each step

    Returns
    -------
    states : array (horizon+1, n_state)  — includes x0
    """
    states = [np.asarray(x0, dtype=np.float32)]
    xk = np.asarray(x0, dtype=np.float32)
    for theta in theta_seq:
        xk = predict_step(learner, xk, np.asarray(theta, dtype=np.float32))
        states.append(xk)
    return np.stack(states, axis=0)
