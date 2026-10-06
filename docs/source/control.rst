Model Predictive Control
========================

ADAlib uses a trained operator as the prediction model inside a
receding-horizon controller (``adalib.run_mpc``). Because the operator is
differentiable and batchable, the controller can use exact
automatic-differentiation gradients (``MPCOptions(gradient="autodiff")``) or
sampling-based optimizers that evaluate many candidate input sequences in one
batched call (``MPCOptions(optimizer="cem")`` or ``"mppi"``).

Runnable examples are in the repository's ``examples/mpc/`` folder:

* ``cstr_tracking_mpc.py`` — CSTR reactor-temperature tracking
* ``triple_tank_tracking_mpc.py`` — triple-tank level tracking
* ``bioreactor_economic_mpc.py`` — fed-batch bioreactor economic MPC
* ``surrogate_mpc_showcase.py`` — finite-difference vs. autodiff gradients and
  batched CEM on the same surrogate

and ``examples/simple_api/04_mpc_example.py`` /
``05_generic_tracking_mpc.py`` show the minimal API for built-in and
user-defined systems. See the project README for the full option list.
