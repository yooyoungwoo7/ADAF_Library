User Implementation
===================

ADALib is implemented as a user-oriented Python library for solving and utilizing
systems of ordinary differential equations (ODEs) within a unified computational
framework.

The library provides four major computational features:

- Forward solving
- Inverse parameter identification
- Operator learning
- Model predictive control (MPC)

Although these features address different numerical tasks, they share a common
implementation philosophy and a consistent user interface.


Unified User Interface
----------------------

The central design principle of ADALib is to separate the mathematical definition
of the dynamical system from the numerical and training configuration used to
solve it.

A user defines the governing ODE system once through a callable system interface
and specifies the numerical or training settings separately through option objects.
Because the system definition and solver configuration are independent, the same
ODE system can be reused across forward solving, inverse identification, operator
learning, and MPC simply by pairing it with different option objects.

The common user schema is illustrated below.

.. code-block:: python

   import adalib

   system  = adalib.CallableODESystem(...)
   options = adalib.FeatureOptions(...)

   result = adalib.run_feature(system, options, ...)

   result.plot()              # or: solution = result.y


The four computational features therefore follow the same high-level structure:

1. Define the ODE system.
2. Configure feature-specific options.
3. Execute the corresponding high-level interface.
4. Obtain a standardized result object.
5. Perform visualization, inference, or subsequent numerical tasks.


General Workflow
----------------

The general workflow of ADALib is summarized below.

.. list-table:: General workflow of the ADALib solver library
   :header-rows: 1
   :widths: 8 92

   * - Step
     - Description

   * - 1
     - Import ADALib and the required scientific computing libraries.

   * - 2
     - Define the target system of ODEs using the callable system interface,
       including the governing equations, initial conditions, time domain,
       and system parameters.

   * - 3
     - Select the desired computational feature among forward solving,
       inverse parameter identification, operator learning, and MPC.

   * - 4
     - Define the corresponding option object, including the ADA basis,
       temporal segmentation, training configuration, and feature-specific
       settings.

   * - 5
     - Execute the selected feature through the corresponding high-level
       interface: ``run_forward``, ``run_inverse``, ``run_operator``,
       or ``run_mpc``.

   * - 6
     - Obtain the standardized result object containing the predicted
       trajectories and feature-specific outputs such as estimated parameters,
       operator predictions, or optimal control sequences.

   * - 7
     - Perform post-processing, visualization, inference, or subsequent
       numerical tasks using the returned result object.


User-Adjustable Settings
------------------------

The four computational features share the common workflow described above,
while each feature exposes a different set of user-adjustable parameters.

Forward Solver
~~~~~~~~~~~~~~

The forward solver mainly exposes settings associated with the ADA
representation, temporal discretization, and optimization procedure.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Category
     - Adjustable settings

   * - Problem configuration
     - Initial conditions, time domain, and fixed system parameters.

   * - ADA representation
     - Basis type, basis order, and number of temporal segments.

   * - Training points
     - Number and distribution of collocation points.

   * - Adam optimizer
     - Number of epochs, inner iterations, learning rate, and numerical precision.

   * - Refinement
     - L-BFGS activation and convergence criteria.


Inverse Parameter Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inverse training extends the forward formulation by introducing observation data
and trainable physical parameters.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Category
     - Adjustable settings

   * - Observation data
     - Number of observations, observation times, and selected state variables.

   * - Data generation
     - Noise level and random seed.

   * - Unknown parameters
     - Initial guesses and lower and upper bounds.

   * - Loss configuration
     - Physics-loss weight and data-loss weight.

   * - ADA representation
     - Basis type, basis order, and number of temporal segments.

   * - Training
     - Adam and L-BFGS settings.


Operator Learning
~~~~~~~~~~~~~~~~~

The operator-learning feature additionally allows the user to define the training
configuration space and the operator-network architecture.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Category
     - Adjustable settings

   * - Input configuration
     - Ranges of system parameters, initial conditions, and external inputs.

   * - Training dataset
     - Number of sampled configurations, sampling strategy, and batch size.

   * - ADA representation
     - Basis type, basis order, and number of temporal segments.

   * - Operator network
     - Hidden layers, layer width, and activation function.

   * - Sequential inputs
     - Number of input or control segments.

   * - Training
     - Number of epochs, learning rate, and optimizer settings.


Model Predictive Control
~~~~~~~~~~~~~~~~~~~~~~~~

The MPC feature provides settings related to prediction, control discretization,
objective definition, constraints, and numerical optimization.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Category
     - Adjustable settings

   * - Prediction
     - Prediction horizon and number of control segments.

   * - Control inputs
     - Initial control sequence and lower and upper control bounds.

   * - Tracking objective
     - Reference trajectory, tracking weights, and control penalty weights.

   * - Economic objective
     - User-defined process-performance objective.

   * - Constraints
     - State and control constraints.

   * - Optimization
     - Maximum iterations, optimizer settings, and convergence criteria.


Feature-Specific Interfaces
---------------------------

Once the system and options are defined, each computational feature can be
executed through its corresponding high-level interface.

.. code-block:: python

   # Forward solving
   forward_result = adalib.run_forward(system, forward_options)

   # Inverse parameter identification
   inverse_result = adalib.run_inverse(system, inverse_options)

   # Operator learning
   operator_result = adalib.run_operator(system, operator_options)

   # Model predictive control
   mpc_result = adalib.run_mpc(system, mpc_options)


All interfaces return structured result objects so that the numerical execution
and subsequent analysis remain separated from the original ODE definition.
