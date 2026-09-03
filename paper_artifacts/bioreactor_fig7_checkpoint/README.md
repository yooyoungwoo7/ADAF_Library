# Bioreactor operator checkpoint — Figure 7 / Table 6 provenance

This directory makes `ADA_paper.tex`'s **Figure 7** and **Table 6**
(`tbl:op_bio`, Section 4.3.4, fed-batch bioreactor operator results)
reproducible from the actual training artifact that produced them,
rather than only from a description.

## Why this exists

The fed-batch bioreactor operator benchmark in the paper was trained
under a "severe-inhibition" parameter regime (`SIN_RANGE=(180,220)`,
`SS0_RANGE=(0,5)`) that predates a later rescale of the *public*
`adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/
bioreactor_problem.py` to a substrate-limited regime
(`SIN_RANGE=(0.3,1.5)`) used for the economic-MPC application in
Section 5.2. The public repository's current sampling ranges cannot
reproduce Figure 7 / Table 6 on their own. This directory archives the
original training run so they can be reproduced exactly instead.

## Contents

| File | What it is |
|---|---|
| `checkpoints/epoch_01997_best.weights.h5` | The trained operator network's weights (Keras `save_weights` format), from the run that produced Table 6's numbers. |
| `config_snapshot.json` | The exact architecture/training config for that run (matches `ADA_paper.tex` Table 15's bioreactor row: `N_p=30`, `hidden=128`, `n_layers=3`, `epochs=2000`, `N_train=50000`, etc.), plus the input normalization stats (`x_mean`/`x_std`) needed to reload the network. |
| `train_summary.json` | Training summary; identifies `epoch_01997` as the best checkpoint by validation physics loss. |
| `bioreactor_problem_snapshot.py` | The exact `BioreactorProblem` class (governing equations, sampling ranges, `output_scale`) as it existed when this checkpoint was trained. **This is an archival snapshot, not the currently-active problem definition** — see the note in that file's docstring area and `RES_SCALE`/`SIN_RANGE` values. |
| `validation_metrics.csv` | Per-case $L_2$ errors from the original evaluation run. Case indices 0, 2, 3, 4 correspond exactly to Table 6's Cases 1–4 (verified to match Table 6's published values to the last reported digit); case index 1 is the excluded high-biomass outlier discussed in the Note under Table 6 ($L_2=9.813\times10^{-4}$). |
| `rollout_data/case_{000,002,003,004}_rollout.npz` | The exact saved `{x_input, t, x_pred, x_ref}` arrays behind Table 6's four reported cases and Figure 7's four panels. Recomputing $\lVert x_{\text{pred}}-x_{\text{ref}}\rVert / \lVert x_{\text{ref}}\rVert$ from these arrays reproduces Table 6's numbers exactly. |
| `regenerate_figure7.py` | Self-contained script that reads `rollout_data/*.npz` and reproduces `ADA_paper.tex`'s Figure 7 (`paper_media/media/image7_v2.png`) in the paper's house plotting style. Requires only `numpy` + `matplotlib` — no TensorFlow, no checkpoint loading needed for this particular figure. |

## Reproducing Figure 7

```bash
cd paper_artifacts/bioreactor_fig7_checkpoint
python regenerate_figure7.py
```

This prints each case's $L_2$ total error (matching `validation_metrics.csv` /
Table 6) and writes `figure7_regenerated.png`, byte-equivalent in content
to `paper_media/media/image7_v2.png`.

## Reloading the checkpoint itself (optional, for verification)

To go one level deeper and re-run inference from the raw network
weights (rather than the saved `rollout_data/*.npz`), you need the
matching model-construction code (`OperatorNet`, `OperatorLearner`,
LPA basis) from the same snapshot as `bioreactor_problem_snapshot.py`.
The architecture is unchanged from the current public
`adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/models/`
modules, so in practice:

1. Temporarily use `bioreactor_problem_snapshot.py` in place of the
   active `problems/bioreactor_problem.py` (do **not** overwrite the
   active file — it now correctly reflects the current, rescaled
   substrate-limited regime used elsewhere in the paper, per
   `.ai/revision_log.md`'s M4 entries).
2. Build `OperatorNet(input_dim=7, state_dim=4, N_p=30, hidden=128,
   n_layers=3, x_mean=..., x_std=..., derived_fn=problem.derived_features_tf,
   n_derived=6, derived_mean=problem.derived_mean, derived_std=problem.derived_std,
   output_scale=problem.output_scale)` using the values in
   `config_snapshot.json` and the snapshot problem class.
3. Call the network once on a dummy input to build its weights, then
   `net.load_weights("checkpoints/epoch_01997_best.weights.h5")`.

This was verified to work (bit-for-bit, up to floating-point
determinism) when this checkpoint was recovered; see
`.ai/revision_log.md` (search "bioreactor_20260427_222342") for the
full provenance investigation and the exact commands used.

## Why this can't be loaded through the public `adalib` API

`adalib` is pip-installable and `import adalib` / `adalib.run_operator(...)`
work today for every system registered in `adalib.get_system(...)`,
including `"fedbatch_bioreactor"`. That is not the issue here.

The issue is that **this specific checkpoint's network was built by a
different, more detailed implementation than the one behind the public
`adalib.systems.FedBatchBioreactor` wrapper**, so its weight tensors
don't match the shapes the public wrapper would construct:

- The public `FedBatchBioreactor`
  (`adalib/systems/fedbatch_bioreactor.py`) and the generic
  `CallableODESystem` it's built on
  (`adalib/systems/callable_system.py`) expose no
  `derived_features_tf` / `output_scale` / `n_derived_features` hooks.
  The checkpoint here was trained with 6 extra physics-derived input
  features (`derived_fn`, see `config_snapshot.json`) and a non-trivial
  per-state `output_scale`, both supplied only by the internal
  `adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/
  bioreactor_problem.py` (`BioreactorProblem`, snapshotted here as
  `bioreactor_problem_snapshot.py`). Building the network through the
  public wrapper omits these, giving a different input dimension —
  `net.load_weights(...)` would fail on a shape mismatch, not run with
  silently wrong results.
- Separately, the public wrapper models `inp` as a **control**
  (`control_names=["inp"]`), while the legacy class this checkpoint was
  trained against treats it as a **parameter** — a structural
  difference independent of the point above.

So this is not a documentation gap or an `OperatorOptions.
reuse_existing_checkpoint` wiring problem (that flag and its
`*best*.weights.h5` checkpoint-file matching already work as expected);
it is a genuine feature gap between the public `FedBatchBioreactor` and
the internal `BioreactorProblem` used for this benchmark. Closing it
would mean extending the public system class with the derived-feature
and output-scale hooks (and reconciling the control-vs-parameter
treatment of `inp`) — a library feature addition, out of scope for
reproducing Figure 7 / Table 6, which is already fully achieved above
via the internal legacy path and the saved `rollout_data/*.npz`.

## Provenance

Recovered from a local training backup (`operator_lib (3).zip`,
run directory `bioreactor_20260427_222342`, dated 2026-04-27) provided
by the paper's author, after the public repository's
`bioreactor_problem.py` had already been rescaled to a different
sampling regime and could no longer reproduce Figure 7 / Table 6 on
its own. Cross-checked against `ADA_paper.tex` Table 15 and against
Table 6's own published numbers before being accepted as the correct
artifact (see `.ai/revision_log.md`).
