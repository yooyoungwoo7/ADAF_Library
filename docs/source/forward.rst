Forward Problem Examples
========================

The ADALib forward solver provides a common interface for solving
user-defined systems of ordinary differential equations.

For each problem, the user follows the same implementation workflow:

1. Define the governing ODE system.
2. Provide the numerical right-hand-side function.
3. Provide the physics-informed residual function.
4. Construct a ``CallableODESystem``.
5. Configure ``ForwardOptions``.
6. Execute ``adalib.run_forward``.
7. Access and visualize the returned result object.

The mathematical definition of the dynamical system is therefore separated
from the numerical and training configuration of the ADA solver.

The following examples demonstrate this workflow for several systems with
different numbers of states and nonlinear structures.


Lotka–Volterra System
---------------------

The Lotka–Volterra example demonstrates a two-state nonlinear
predator–prey system.

.. toctree::
   :maxdepth: 1

   ex_lotka


Euler Rigid-Body System
-----------------------

The Euler example demonstrates a three-state coupled nonlinear dynamical
system describing torque-free rigid-body rotation.

.. toctree::
   :maxdepth: 1

   ex_euler


Damped Nonlinear Pendulum
-------------------------

The pendulum example demonstrates how a second-order differential equation
can be reformulated as a first-order system and implemented through
``CallableODESystem``.

.. toctree::
   :maxdepth: 1

   ex_pendulum
