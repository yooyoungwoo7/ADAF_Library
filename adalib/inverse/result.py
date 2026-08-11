"""
adalib/inverse/result.py

InverseResult — result object returned by run_inverse().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class InverseResult:
    """Result of a physics-informed inverse training run.

    Attributes
    ----------
    t : ndarray, shape (Nt_total,)
        Time grid.
    y : ndarray, shape (n_state, Nt_total)
        Recovered trajectory (same layout as ForwardResult.y).
    estimated_params : dict[str, float]
        Final constrained values of each InverseParameter.
    parameter_history : dict[str, list of float]
        Per-step parameter snapshots (recorded every ``param_log_every`` Adam steps).
    loss_history : list of float
        Total loss at the end of each segment.
    physics_loss_history : list of float
        Physics component of the loss, per segment.
    data_loss_history : list of float
        Data component of the loss, per segment.
    runtime_sec : float
        Wall-clock elapsed time.
    metadata : dict
        System name, t_span, x0, and option fields.
    """

    def __init__(
        self,
        raw_output: Dict,
        inverse_params: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ):
        raw_traj = raw_output["trajectory"]   # (Nt_total, n_state)
        self._t   = raw_output["t"]           # (Nt_total,)
        self._y   = raw_traj.T                # (n_state, Nt_total)

        from .parameter import InverseParameter
        self.estimated_params: Dict[str, float] = {}
        for name, ip in inverse_params.items():
            if isinstance(ip, InverseParameter):
                self.estimated_params[name] = ip.numpy_value
            else:
                self.estimated_params[name] = float(ip)

        self.parameter_history:    Dict[str, List[float]] = raw_output.get("param_history", {})
        self.loss_history:         List[float]            = raw_output.get("loss_history", [])
        self.physics_loss_history: List[float]            = raw_output.get("physics_loss_history", [])
        self.data_loss_history:    List[float]            = raw_output.get("data_loss_history", [])
        self.runtime_sec:          float                  = raw_output.get("runtime_sec", 0.0)
        self.metadata:             Dict                   = metadata or {}

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def y(self) -> np.ndarray:
        return self._y

    def to_arrays(self):
        """Return ``(t, y)`` as plain numpy arrays."""
        return np.asarray(self._t), np.asarray(self._y)

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot(
        self,
        state_names=None,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
        observation_data=None,
        true_params: Optional[Dict[str, float]] = None,
    ):
        """Plot the recovered trajectory and parameter convergence.

        Parameters
        ----------
        state_names : list of str, optional
            Labels for state axes.
        save_path : str, optional
            Path prefix.  Two files are saved:
            ``<save_path>_trajectory.png`` and ``<save_path>_params.png``.
        show : bool
            Call ``plt.show()`` after each figure.
        title : str, optional
            Trajectory figure suptitle.
        observation_data : ObservationData, optional
            If provided, overlay scattered observations on the trajectory.
        true_params : dict[str, float], optional
            Known true parameter values.  When supplied, draws a horizontal
            dashed line per parameter so convergence can be judged visually.
            Example: ``{"alpha": 2.0, "gamma": 1.06}``

        Returns
        -------
        (fig_traj, fig_params)
        """
        fig_t   = self._plot_trajectory(
            state_names=state_names,
            save_path=save_path + "_trajectory.png" if save_path else None,
            show=show,
            title=title,
            observation_data=observation_data,
        )

        fig_p = self.plot_params(
            true_params=true_params,
            save_path=save_path + "_params.png" if save_path else None,
            show=show,
        )

        return fig_t, fig_p

    def plot_params(
        self,
        true_params: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
        figsize=None,
    ):
        """Plot parameter convergence curves (paper Fig. 9 style).

        All inverse parameters are shown on a single axis.  Each parameter
        gets a distinct color.  If ``true_params`` is provided, horizontal
        dashed lines mark the ground-truth values for easy comparison.

        Parameters
        ----------
        true_params : dict[str, float], optional
            ``{"alpha": 2.0, "gamma": 1.06}`` — drawn as dashed reference lines.
        save_path : str, optional
            File path for saving (PNG / PDF).
        show : bool
            Call ``plt.show()`` when True.
        title : str, optional
            Axes title.

        Returns
        -------
        (fig, ax)  or  None if parameter_history is empty
        """
        if not self.parameter_history or all(
            len(v) == 0 for v in self.parameter_history.values()
        ):
            return None

        from ..utils.plotting import plot_inverse_params
        return plot_inverse_params(
            param_history    = self.parameter_history,
            true_params      = true_params,
            estimated_params = self.estimated_params,
            save_path        = save_path,
            show             = show,
            title            = title,
            figsize          = figsize,
        )

    def plot_loss(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plot total / physics / data loss per segment.

        Returns
        -------
        fig
        """
        from ..utils.plotting import _STATE_COLORS
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        segs = np.arange(len(self.loss_history))

        ax.semilogy(segs, self.loss_history,         color="k",          lw=1.8, label="total")
        ax.semilogy(segs, self.physics_loss_history, color=_STATE_COLORS[0], lw=1.5,
                    ls="--", label="physics")
        ax.semilogy(segs, self.data_loss_history,    color=_STATE_COLORS[1], lw=1.5,
                    ls=":",  label="data")

        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title("Inverse training loss")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    # ── Internal helpers ──────────────────────────────────────────────────

    def _plot_trajectory(
        self,
        state_names=None,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
        observation_data=None,
    ):
        from ..utils.plotting import _C_PRED, _C_REF, _tight_x
        import matplotlib.pyplot as plt

        n_state = self._y.shape[0]
        names   = state_names or [f"$x_{{{j}}}$" for j in range(n_state)]

        fig, axes = plt.subplots(
            n_state, 1,
            figsize=(7, 2.8 * n_state),
            squeeze=False,
            sharex=True,
        )

        for j, ax in enumerate(axes[:, 0]):
            lbl = names[j] if j < len(names) else f"$x_{{{j}}}$"
            ax.plot(self._t, self._y[j], color=_C_PRED, lw=1.8,
                    label="ADA inverse")

            if observation_data is not None and j in observation_data.state_indices:
                col = observation_data.state_indices.index(j)
                ax.scatter(
                    observation_data.t,
                    observation_data.y[:, col],
                    s=12, color=_C_REF, zorder=3, label="observations",
                )

            ax.set_ylabel(lbl)
            if j == 0:
                ax.legend(fontsize=9, loc="best")
            _tight_x(ax, self._t)

        axes[-1, 0].set_xlabel("$t$")
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    # ── I/O ───────────────────────────────────────────────────────────────

    def save_npz(self, path: Optional[str] = None) -> str:
        """Save ``t``, ``y``, and ``estimated_params`` to a compressed .npz file."""
        if path is None:
            path = "inverse_result.npz"
        np.savez_compressed(
            path,
            t=self._t,
            y=self._y,
            **{f"param_{k}": np.array(v) for k, v in self.estimated_params.items()},
        )
        return path

    def save_all(
        self,
        output_dir: str,
        state_names=None,
        true_params: Optional[Dict[str, float]] = None,
        observation_data=None,
        prefix: Optional[str] = None,
    ) -> str:
        """Save all run outputs to ``output_dir``.

        Saves:
          - ``<prefix>_trajectory.npz``  — t, y, estimated params
          - ``<prefix>_loss_history.csv``  — step, total, physics, data
          - ``<prefix>_param_history.csv`` — step, param values
          - ``<prefix>_trajectory.png``
          - ``<prefix>_params.png``
          - ``<prefix>_loss.png``
          - ``run_metadata.json``

        Parameters
        ----------
        output_dir : str
            Directory to write into (created if needed).
        state_names : list of str, optional
            Overrides metadata state names for plot labels.
        true_params : dict, optional
            Known true values; draws reference lines on param plot.
        observation_data : ObservationData, optional
            Overlays scattered observations on trajectory plot.
        prefix : str, optional
            File name prefix.  Defaults to the system name from metadata.

        Returns
        -------
        str  — the resolved ``output_dir`` path.
        """
        import os, json
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        sys_name = self.metadata.get("system_name", "inverse")
        p = prefix or sys_name
        snames = state_names or self.metadata.get("state_names") or None

        # ── Trajectory NPZ ────────────────────────────────────────
        self.save_npz(os.path.join(output_dir, f"{p}_trajectory.npz"))

        # ── Loss history CSV ──────────────────────────────────────
        loss_path = os.path.join(output_dir, f"{p}_loss_history.csv")
        with open(loss_path, "w", newline="") as f:
            f.write("step,total_loss,physics_loss,data_loss\n")
            for i, (tl, pl, dl) in enumerate(zip(
                self.loss_history,
                self.physics_loss_history,
                self.data_loss_history,
            )):
                f.write(f"{i},{tl},{pl},{dl}\n")

        # ── Param history CSV ─────────────────────────────────────
        if self.parameter_history and any(
            len(v) for v in self.parameter_history.values()
        ):
            param_names = list(self.parameter_history.keys())
            n_rows = max(len(v) for v in self.parameter_history.values())
            ph_path = os.path.join(output_dir, f"{p}_param_history.csv")
            with open(ph_path, "w", newline="") as f:
                f.write("step," + ",".join(param_names) + "\n")
                for i in range(n_rows):
                    row = [str(i)]
                    for nm in param_names:
                        vals = self.parameter_history[nm]
                        row.append(f"{vals[i]:.8g}" if i < len(vals) else "")
                    f.write(",".join(row) + "\n")

        # ── Plots ──────────────────────────────────────────────────
        fig_t = self._plot_trajectory(
            state_names=snames,
            save_path=os.path.join(output_dir, f"{p}_trajectory.png"),
            observation_data=observation_data,
        )
        plt.close(fig_t)

        fig_p = self.plot_params(
            true_params=true_params,
            save_path=os.path.join(output_dir, f"{p}_params.png"),
        )
        if fig_p is not None:
            plt.close(fig_p[0])

        fig_l = self.plot_loss(
            save_path=os.path.join(output_dir, f"{p}_loss.png"),
        )
        plt.close(fig_l)

        # ── Metadata JSON ─────────────────────────────────────────
        meta_out = {
            **self.metadata,
            "runtime_sec":      self.runtime_sec,
            "estimated_params": self.estimated_params,
            "n_steps":          len(self.loss_history),
            "final_loss":       self.loss_history[-1] if self.loss_history else None,
        }
        with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

        return output_dir

    # ── Display ───────────────────────────────────────────────────────────

    def __repr__(self):
        params_str = ", ".join(f"{k}={v:.5g}" for k, v in self.estimated_params.items())
        return (
            f"InverseResult(\n"
            f"  estimated_params={{{params_str}}},\n"
            f"  n_steps={len(self.loss_history)},\n"
            f"  final_loss={self.loss_history[-1] if self.loss_history else 'N/A'},\n"
            f"  runtime={self.runtime_sec:.1f}s\n"
            f")"
        )
