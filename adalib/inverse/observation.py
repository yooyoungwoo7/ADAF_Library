"""
adalib/inverse/observation.py

ObservationData — standardized container for inverse problem measurements.
data_gen        — extract observations from a ForwardResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class ObservationData:
    """Standardized observation container for inverse problems.

    Parameters
    ----------
    t : array-like, shape (N_obs,)
        Observation times, sorted ascending.
    y : array-like, shape (N_obs, n_observed)
        Observed state values.  Columns correspond to ``state_indices``.
    state_indices : list of int
        Which ODE state variables are observed.  ``y[:, k]`` is the
        measurement for ``state_indices[k]``.

    Both ``data_gen()`` and direct construction produce the same interface,
    so ``run_inverse`` never needs to distinguish the source.

    Examples
    --------
    Synthetic data from ForwardResult:

    >>> data = adalib.data_gen(result, n_points=100, noise_std=0.01, seed=42)

    External experimental data:

    >>> data = ObservationData(t=t_exp, y=y_exp, state_indices=[0, 1])
    """

    t: np.ndarray
    y: np.ndarray
    state_indices: List[int]

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        N = len(self.t)
        if N == 0:
            raise ValueError("ObservationData: t is empty.")

        if self.y.shape[0] != N:
            raise ValueError(
                f"ObservationData: len(t)={N} but y.shape[0]={self.y.shape[0]}."
            )

        if len(self.state_indices) == 0:
            raise ValueError("ObservationData: state_indices is empty.")

        if self.y.shape[1] != len(self.state_indices):
            raise ValueError(
                f"ObservationData: y has {self.y.shape[1]} columns but "
                f"state_indices has {len(self.state_indices)} entries."
            )

        if not np.all(np.isfinite(self.t)):
            raise ValueError("ObservationData: t contains non-finite values.")

        if not np.all(np.isfinite(self.y)):
            raise ValueError("ObservationData: y contains non-finite values.")

        if len(self.t) > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError(
                "ObservationData: t must be strictly increasing (no duplicates)."
            )

        for idx in self.state_indices:
            if not isinstance(idx, (int, np.integer)) or idx < 0:
                raise ValueError(
                    f"ObservationData: state_indices must be non-negative ints; "
                    f"got {idx!r}."
                )

    # ------------------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return len(self.t)

    @property
    def n_observed_states(self) -> int:
        return len(self.state_indices)

    def time_range(self):
        """Return (t_min, t_max)."""
        return float(self.t[0]), float(self.t[-1])

    def __repr__(self):
        return (
            f"ObservationData(n_obs={self.n_obs}, "
            f"state_indices={self.state_indices}, "
            f"t=[{self.t[0]:.4g}, {self.t[-1]:.4g}])"
        )


# ---------------------------------------------------------------------------

def data_gen(
    result,
    n_points: Optional[int] = None,
    times: Optional[np.ndarray] = None,
    state_indices: Optional[List[int]] = None,
    noise_std: Union[float, Sequence[float]] = 0.0,
    seed: Optional[int] = None,
    source: str = "prediction",
) -> ObservationData:
    """Generate observation data from a ``ForwardResult``.

    This function does **not** run any ODE solver.  Call ``run_forward``
    first; then pass its result here.

    Parameters
    ----------
    result : ForwardResult
        Completed forward solve result (``result.t``, ``result.y``).
    n_points : int, optional
        Number of uniformly-sampled time points.  Mutually exclusive
        with ``times``.
    times : array-like, optional
        Explicit observation times within ``[result.t[0], result.t[-1]]``.
        Mutually exclusive with ``n_points``.
    state_indices : list of int, optional
        Which states to observe.  Defaults to all states.
    noise_std : float or sequence of float
        Standard deviation of additive Gaussian noise.  If a sequence,
        one value per observed state.
    seed : int, optional
        Random seed for reproducible noise.
    source : {"prediction"}
        Which part of the result to sample from.  Currently only
        ``"prediction"`` is supported (uses ``result.y``).

    Returns
    -------
    ObservationData

    Raises
    ------
    ValueError
        On invalid argument combinations or out-of-range observation times.
    """
    # ── Validate source ──────────────────────────────────────────────
    if source not in ("prediction",):
        raise ValueError(
            f"data_gen: source={source!r} is not supported. "
            "Use source='prediction'."
        )

    t_full = np.asarray(result.t, dtype=float).ravel()
    y_full = np.asarray(result.y, dtype=float)
    # y_full shape: (n_state, N_t)  — forward result convention

    n_state = y_full.shape[0]

    # ── State indices ────────────────────────────────────────────────
    if state_indices is None:
        state_indices = list(range(n_state))
    else:
        state_indices = list(state_indices)
        for idx in state_indices:
            if idx < 0 or idx >= n_state:
                raise ValueError(
                    f"data_gen: state_indices contains {idx} but system has "
                    f"{n_state} states (valid range 0..{n_state-1})."
                )

    # ── Time sampling ────────────────────────────────────────────────
    if n_points is not None and times is not None:
        raise ValueError("data_gen: specify at most one of n_points or times.")

    if n_points is not None:
        if n_points < 1:
            raise ValueError(f"data_gen: n_points must be >= 1, got {n_points}.")
        t_obs = np.linspace(t_full[0], t_full[-1], n_points)
    elif times is not None:
        t_obs = np.asarray(times, dtype=float).ravel()
        if np.any(t_obs < t_full[0] - 1e-10) or np.any(t_obs > t_full[-1] + 1e-10):
            raise ValueError(
                f"data_gen: some times fall outside result range "
                f"[{t_full[0]:.6g}, {t_full[-1]:.6g}]."
            )
        t_obs = np.clip(t_obs, t_full[0], t_full[-1])
        if len(t_obs) > 1 and not np.all(np.diff(t_obs) > 0):
            # de-duplicate and sort
            t_obs = np.unique(t_obs)
    else:
        # Default: 50 evenly-spaced points
        t_obs = np.linspace(t_full[0], t_full[-1], 50)

    # ── Interpolate at observation times ─────────────────────────────
    y_obs = np.zeros((len(t_obs), len(state_indices)), dtype=float)
    for col, idx in enumerate(state_indices):
        y_obs[:, col] = np.interp(t_obs, t_full, y_full[idx])

    # ── Add noise ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    noise_std_arr = np.asarray(noise_std, dtype=float)
    if noise_std_arr.ndim == 0:
        noise_std_arr = np.full(len(state_indices), float(noise_std_arr))
    if noise_std_arr.shape != (len(state_indices),):
        raise ValueError(
            f"data_gen: noise_std must be a scalar or length-{len(state_indices)} "
            f"sequence; got shape {noise_std_arr.shape}."
        )
    for col in range(len(state_indices)):
        if noise_std_arr[col] > 0:
            y_obs[:, col] += rng.normal(0.0, noise_std_arr[col], size=len(t_obs))

    return ObservationData(t=t_obs, y=y_obs, state_indices=state_indices)
