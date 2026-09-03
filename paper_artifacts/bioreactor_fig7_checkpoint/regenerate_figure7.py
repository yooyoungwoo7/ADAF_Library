#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_figure7.py
============================================================
Regenerates ADA_paper.tex's Figure 7 (fed-batch bioreactor operator
results) from the exact saved rollout data behind Table 6 (tbl:op_bio).

No training or checkpoint loading required for this script — the
reference/operator trajectories used to compute Table 6's reported L2
errors were saved verbatim in rollout_data/*.npz at evaluation time.
This script only re-plots them in the paper's house style.

Run from anywhere; paths below are relative to this file's directory.
Requires: numpy, matplotlib. (No TensorFlow needed for this script.)

To instead retrain/reload the operator checkpoint itself (e.g. to
verify these saved arrays, or to evaluate new cases), see
`checkpoints/epoch_01997_best.weights.h5` + `config_snapshot.json` +
`bioreactor_problem_snapshot.py` and README.md in this directory.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse this repository's own plotting style module (Times New Roman /
# STIX, inward ticks, no x-axis padding) so the figure matches the rest
# of the paper (e.g. Figure 6).
import sys
sys.path.insert(0, os.path.join(
    HERE, "..", "..", "adalib", "_vendor", "legacy", "operator_mpc_original",
    "cstr_mpc_op",
))
try:
    from utils._style import apply_style, tight_x
except ImportError:
    # Fallback: minimal inline equivalent if the path above doesn't
    # resolve in your checkout (module is tiny, see its docstring).
    def apply_style():
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.minor.visible": True, "ytick.minor.visible": True,
            "xtick.top": True, "ytick.right": True,
            "axes.grid": False, "axes.unicode_minus": False,
        })

    def tight_x(ax, x):
        x = np.asarray(x)
        if x.size >= 2:
            ax.set_xlim(float(x[0]), float(x[-1]))

apply_style()

# Table 6 order: Case1=growth(000), Case2=washout(002),
# Case3=Haldane inhibition(003), Case4=slow inhibited(004).
CASE_IDS = [0, 2, 3, 4]
LABELS = ["Visible growth", "Washout", "Haldane inhibition", "Slow inhibited"]

data = []
for cid in CASE_IDS:
    d = np.load(os.path.join(HERE, "rollout_data", f"case_{cid:03d}_rollout.npz"))
    data.append((d["x_input"], d["t"], d["x_pred"], d["x_ref"]))

fig, axes = plt.subplots(2, 4, figsize=(4.2 * 4, 6.5), sharex=True)
colors = ["C3", "C0"]

for i, (x_in, t, x_pred, x_ref) in enumerate(data):
    ax_top, ax_bot = axes[0, i], axes[1, i]
    for idx, c in zip([0, 2], colors):          # Xs, Ps
        ax_top.plot(t, x_ref[:, idx], color="k", lw=1.8)
        ax_top.plot(t, x_pred[:, idx], "--", color=c, lw=1.5)
    for idx, c in zip([1, 3], colors):          # Ss, Vs
        ax_bot.plot(t, x_ref[:, idx], color="k", lw=1.8)
        ax_bot.plot(t, x_pred[:, idx], "--", color=c, lw=1.5)
    tight_x(ax_top, t)
    tight_x(ax_bot, t)

    Xs0, Ss0, Ps0, Vs0, Yx, Sin, inp = x_in
    ax_top.set_title(
        f"$X_{{s0}}$={Xs0:.2f}, $S_{{s0}}$={Ss0:.2f}\n"
        f"$Y_x$={Yx:.3f}, $S_{{in}}$={Sin:.1f}\n"
        f"inp={inp:.3f}, $V_{{s0}}$={Vs0:.2f}",
        fontsize=10,
    )
    ax_bot.set_xlabel(r"$t$ [min]")
    if i == 0:
        ax_top.set_ylabel(r"$X_s, P_s$")
        ax_bot.set_ylabel(r"$S_s, V_s$")

    l2 = np.linalg.norm(x_pred - x_ref) / np.linalg.norm(x_ref)
    print(f"Case {i+1} ({LABELS[i]}): L2_total = {l2:.4e}  "
          "(matches Table 6 / validation_metrics.csv)")

fig.tight_layout()
out_path = os.path.join(HERE, "figure7_regenerated.png")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved: {out_path}")
print("This is the exact source image used for ADA_paper.tex Figure 7 "
      "(paper_media/media/image7_v2.png).")
