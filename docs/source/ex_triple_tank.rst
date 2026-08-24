Triple-Tank Operator Learning
=============================

Overview
--------

This tutorial demonstrates the ADA-based operator-learning workflow using
the Triple-Tank system as an example.

Although only the Triple-Tank problem is presented in this tutorial, the
same operator-learning workflow can also be applied to other parametric
ODE systems, such as the CSTR and Fed-Batch Bioreactor. In such cases,
the system definition and the corresponding system-specific initial
conditions, inputs, and configurations should be changed accordingly.

For the Triple-Tank benchmark, the system is loaded using

.. code-block:: python

   system = adalib.get_system("triple_tank")


Problem Setup
-------------

The Triple-Tank benchmark consists of three interconnected tanks.

The state variables are

.. math::

   \mathbf{x}
   =
   \begin{bmatrix}
   h_1 & h_2 & h_3
   \end{bmatrix}^{T},

where

- :math:`h_1`: liquid level of Tank 1,
- :math:`h_2`: liquid level of Tank 2,
- :math:`h_3`: liquid level of Tank 3.

The external pump inputs are

.. math::

   \mathbf{q}
   =
   \begin{bmatrix}
   q_1 & q_2
   \end{bmatrix}^{T},

where :math:`q_1` and :math:`q_2` denote the pump flow rates supplied to
Tanks 1 and 2.

The system dynamics are represented as

.. math::

   \frac{dh_1}{dt}
   =
   \frac{q_1-q_{13}}{A},

.. math::

   \frac{dh_2}{dt}
   =
   \frac{q_2-q_{32}}{A},

.. math::

   \frac{dh_3}{dt}
   =
   \frac{q_{13}+q_{32}-q_{30}}{A},

where :math:`q_{13}`, :math:`q_{32}`, and :math:`q_{30}` represent
gravity-driven flows between the tanks and through the outlet.


Operator-Learning Workflow
--------------------------

Unlike the forward solver, which solves one specific ODE configuration,
operator learning trains a reusable model that can predict the ADA
representation for different system configurations.

The overall workflow consists of the following steps:

1. Select the target ODE system.
2. Configure ``OperatorOptions``.
3. Generate the operator-training dataset.
4. Train the operator network.
5. Perform inference for a target configuration.
6. Save and reuse the trained checkpoint.
7. Evaluate new initial conditions and system inputs.
8. Compare the operator predictions with numerical reference solutions.

The workflow can be summarized as

.. code-block:: text

   ODE System
       |
       v
   OperatorOptions
       |
       v
   Training Data Generation
       |
       v
   Operator Training
       |
       v
   Trained Checkpoint
       |
       v
   New Initial Condition / Input
       |
       v
   run_operator
       |
       v
   OperatorResult


1. Import Libraries
~~~~~~~~~~~~~~~~~~~

First, import NumPy, Matplotlib, and ADALib.

.. code-block:: python

   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import adalib

   adalib.utils.set_adalib_plot_style(style="serif")

The Matplotlib ``Agg`` backend is used because the generated figures are
saved directly to files instead of being displayed interactively.


2. Load the Triple-Tank System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Triple-Tank benchmark is provided as a built-in ADALib system.

.. code-block:: python

   system = adalib.get_system("triple_tank")

The resulting system object contains the governing equations required by
the operator-learning workflow.

For other built-in benchmark systems, the same interface can be used by
changing the registered system name and supplying the corresponding
system-specific configuration.


3. Configure Operator Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The operator-learning procedure is configured through
``adalib.OperatorOptions``.

.. code-block:: python

   options = adalib.OperatorOptions(
       basis="lpa",

       # Data generation
       n_train=2000,
       n_val=200,
       seed=42,
       generate_data=True,
       reuse_existing_data=False,

       # Training
       train=True,
       reuse_existing_checkpoint=False,
       epochs=1000,
       batch_size=8,
       lr=3e-3,
       hidden=64,
       n_layers=2,

       # Inference
       infer=True,

       work_dir="./runs/operator_triple_tank",
       verbose=True,
   )

The principal settings are:

- ``basis``: ADA basis representation used for the operator output.
- ``n_train``: number of training configurations.
- ``n_val``: number of validation configurations.
- ``seed``: random seed used during dataset generation.
- ``generate_data``: enables generation of a new training dataset.
- ``reuse_existing_data``: determines whether an existing dataset is reused.
- ``train``: enables operator-network training.
- ``reuse_existing_checkpoint``: determines whether a previously trained model is loaded.
- ``epochs``: number of training epochs.
- ``batch_size``: number of configurations processed in each training batch.
- ``lr``: learning rate.
- ``hidden``: width of the hidden layers.
- ``n_layers``: number of hidden layers.
- ``infer``: enables operator inference after training.
- ``work_dir``: directory used to store generated data, checkpoints, and results.

In this example, the LPA basis is used for the ADA representation.


4. Train the Operator
~~~~~~~~~~~~~~~~~~~~~

The first call to ``adalib.run_operator`` performs the complete
data-generation, training, and inference workflow.

.. code-block:: python

   result = adalib.run_operator(
       system=system,
       x0=[40.0, 20.0, 30.0],
       t_span=(0.0, 0.5),
       params=[100.0, 150.0],
       options=options,
   )

The initial tank levels are

.. math::

   \mathbf{x}_0
   =
   [40,\ 20,\ 30],

and the corresponding pump inputs are

.. math::

   [q_1,\ q_2]
   =
   [100,\ 150].

The returned ``OperatorResult`` contains the predicted state trajectory
for the supplied configuration.


5. Inspect the OperatorResult
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The predicted time coordinates and state trajectories are directly
available from the returned result object.

.. code-block:: python

   print("\n=== OperatorResult (Case 1) ===")
   print(f"t shape  : {result.t.shape}")
   print(f"y shape  : {result.y.shape}")
   print(
       f"t range  : "
       f"[{result.t[0]:.2f}, {result.t[-1]:.2f}]"
   )

The principal output arrays are

- ``result.t``: predicted time coordinates,
- ``result.y``: predicted state trajectories.

For the Triple-Tank system,

.. code-block:: text

   result.y[0] -> h1
   result.y[1] -> h2
   result.y[2] -> h3


6. Reuse the Trained Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the operator has been trained, data generation and training do not
need to be repeated for new system configurations.

An inference-only option object is therefore defined as

.. code-block:: python

   options_infer = adalib.OperatorOptions(
       basis="lpa",
       generate_data=False,
       train=False,
       reuse_existing_checkpoint=True,
       infer=True,
       work_dir="./runs/operator_triple_tank",
       hidden=64,
       n_layers=2,
       verbose=False,
   )

The important differences from the training configuration are

.. code-block:: text

   generate_data = False
   train = False
   reuse_existing_checkpoint = True

The trained operator stored in

.. code-block:: text

   ./runs/operator_triple_tank

is therefore loaded and reused directly.

This separation between offline training and subsequent inference is a
key feature of the operator-learning workflow.


7. Define Test Cases
~~~~~~~~~~~~~~~~~~~~

Three combinations of initial tank levels and pump inputs are used to
evaluate the trained operator.

.. code-block:: python

   TEST_CASES = [
       {
           "x0": [40.0, 20.0, 30.0],
           "params": [100.0, 150.0],
       },
       {
           "x0": [25.0, 45.0, 35.0],
           "params": [80.0, 200.0],
       },
       {
           "x0": [50.0, 15.0, 10.0],
           "params": [60.0, 100.0],
       },
   ]

Each test case contains

- ``x0``: the three initial tank levels,
- ``params``: the two pump-flow inputs.


8. Run Inference for New Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first test case has already been evaluated during the initial
training call.

The remaining cases are evaluated using the trained checkpoint without
additional operator training.

.. code-block:: python

   all_results = [result]

   for tc in TEST_CASES[1:]:
       r = adalib.run_operator(
           system=system,
           x0=tc["x0"],
           t_span=(0.0, 0.5),
           params=tc["params"],
           options=options_infer,
       )

       all_results.append(r)

The same trained operator is therefore reused for multiple configurations.

.. code-block:: text

   Trained Operator
          |
          +-- Case 1 --> OperatorResult
          |
          +-- Case 2 --> OperatorResult
          |
          +-- Case 3 --> OperatorResult

No additional training is required for Cases 2 and 3.


9. Multi-Case Numerical Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The operator predictions are compared with numerical reference
trajectories for all three test cases.

First, the state names and corresponding plot labels are defined.

.. code-block:: python

   state_names = [
       "h1",
       "h2",
       "h3",
   ]

   state_labels = [
       "$h_1$ [cm]",
       "$h_2$ [cm]",
       "$h_3$ [cm]",
   ]

Labels describing the initial conditions and pump inputs of the
individual test cases are then generated.

.. code-block:: python

   col_labels = []

   for tc in TEST_CASES:
       h = tc["x0"]
       q = tc["params"]

       col_labels.append(
           f"$h_1$={h[0]:.0f}, "
           f"$h_2$={h[1]:.0f}, "
           f"$h_3$={h[2]:.0f} cm\n"
           f"$q_1$={q[0]:.0f}, "
           f"$q_2$={q[1]:.0f} cm³/s"
       )

The initial conditions and pump inputs are collected into lists for the
numerical reference calculation.

.. code-block:: python

   x0_list = [
       tc["x0"]
       for tc in TEST_CASES
   ]

   ctrl_list = [
       tc["params"]
       for tc in TEST_CASES
   ]

The three operator predictions are compared with SciPy ``solve_ivp``
using ``adalib.utils.plot_operator_result``.

.. code-block:: python

   fig2, axes2, metrics2 = adalib.utils.plot_operator_result(
       all_results,
       system=system,
       x0=x0_list,
       control=ctrl_list,
       reference="solve_ivp",
       state_names=state_labels,
       labels=col_labels,
       state_groups=[[0], [1], [2]],
       title=(
           "Three-Tank Benchmark — "
           "Operator vs Reference (3 cases)"
       ),
       save_path="triple_tank_operator_3cases.png",
       show=False,
   )

The relative :math:`L_2` error is evaluated independently for each state
and each test configuration.

.. math::

   \varepsilon_i
   =
   \frac{
   \left\|
   y_i^{\mathrm{Operator}}
   -
   y_i^{\mathrm{ref}}
   \right\|_2
   }{
   \left\|
   y_i^{\mathrm{ref}}
   \right\|_2
   }.

The errors can be printed using

.. code-block:: python

   print("L2 rel errors (per case, per state):")

   for i, row in enumerate(metrics2["l2_rel"]):
       print(
           f"Case {i+1}: "
           + ", ".join(
               f"{n}={v:.2e}"
               for n, v in zip(state_names, row)
           )
       )

The resulting comparison between the trained operator and the numerical
reference solutions is shown below.

.. figure:: triple_tank_operator_3cases.png
   :width: 95%
   :align: center
   :alt: Triple-Tank operator comparison for three test cases


Training and Inference Summary
------------------------------

The complete operator-learning workflow can be summarized as follows.

.. code-block:: text

   First Execution
   ---------------

   Triple-Tank System
          |
          v
   Generate Training Data
          |
          v
   Train LPA Operator
          |
          v
   Save Checkpoint


   Subsequent Executions
   ---------------------

   New Initial Condition / Pump Inputs
          |
          v
   Load Existing Checkpoint
          |
          v
   Operator Inference
          |
          v
   OperatorResult

The computationally expensive training stage is performed only once.

The trained operator can subsequently be reused for different initial
conditions and pump-input configurations without retraining.


Complete Source Code
--------------------

The tutorial above focuses on the multi-case numerical comparison to
avoid redundant visualization.

The complete runnable test module also contains additional helper
visualizations for single-case comparison and inference validation.

.. literalinclude:: ../../tests/test_adalib_operator_triple_tank.py
   :language: python
   :linenos:
