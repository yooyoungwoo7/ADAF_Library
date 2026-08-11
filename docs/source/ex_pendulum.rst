Damped Nonlinear Pendulum
=========================

Problem Setup
-------------

This example demonstrates the implementation of a damped nonlinear
pendulum as a user-defined ODE system in ADALib.

The governing second-order equation is

.. math::

   \frac{d^2\theta}{dt^2}
   +
   \gamma\frac{d\theta}{dt}
   +
   \frac{g}{L}\sin\theta
   =
   0,

where :math:`\theta` denotes the angular displacement.

Introducing the angular velocity

.. math::

   \omega=\frac{d\theta}{dt},

the equation is converted into the first-order system

.. math::

   \frac{d\theta}{dt}
   =
   \omega,

.. math::

   \frac{d\omega}{dt}
   =
   -\gamma\omega
   -
   \frac{g}{L}\sin\theta.

The parameter values are

.. math::

   \gamma=0.30~\mathrm{s}^{-1},

.. math::

   \frac{g}{L}=9.81~\mathrm{s}^{-2}.

The pendulum is released from an angular displacement of 60 degrees with
zero initial angular velocity,

.. math::

   \theta(0)=\frac{\pi}{3},
   \qquad
   \omega(0)=0,

and is solved over

.. math::

   t\in[0,10]~\mathrm{s}.


Implementation
--------------

The implementation follows the same common forward-solving interface used
for the other user-defined systems.


1. Import Libraries
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import adalib


2. Define the System Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The damping coefficient and gravitational parameter are

.. code-block:: python

   GAMMA = 0.30
   G_OVER_L = 9.81


3. Define the Numerical RHS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conventional first-order form of the ODE is defined through a Python
callable.

.. code-block:: python

   def pendulum_rhs(t, state, u=None, p=None):
       theta, omega = state

       return [
           omega,
           -GAMMA * omega
           - G_OVER_L * np.sin(theta),
       ]

The first component represents

.. math::

   \dot{\theta}=\omega,

while the second component represents

.. math::

   \dot{\omega}
   =
   -\gamma\omega
   -
   \frac{g}{L}\sin\theta.


4. Define the Physics Residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For physics-informed optimization, the same system is represented in
residual form.

.. code-block:: python

   def pendulum_rhs_tf(var_list, i, u=None, p=None):
       import tensorflow as tf

       theta, theta_t = var_list[0]
       omega, omega_t = var_list[1]

       if i == 0:
           return theta_t - omega

       else:
           return omega_t - (
               -GAMMA * omega
               - G_OVER_L * tf.sin(theta)
           )

The corresponding residuals are

.. math::

   \mathcal{R}_1
   =
   \dot{\theta}-\omega,

.. math::

   \mathcal{R}_2
   =
   \dot{\omega}
   +
   \gamma\omega
   +
   \frac{g}{L}\sin\theta.


5. Construct CallableODESystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two system representations are combined through
``CallableODESystem``.

.. code-block:: python

   system = adalib.CallableODESystem(
       name="damped_pendulum",
       rhs=pendulum_rhs,
       rhs_tf=pendulum_rhs_tf,
       state_names=["theta", "omega"],
   )

The state ordering defined in ``state_names`` is retained throughout the
forward-solving workflow.


6. Configure ForwardOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ADA representation, temporal segmentation, and optimizer settings are
specified separately from the governing equations.

.. code-block:: python

   options = adalib.ForwardOptions(
       basis="adaf",
       n_seg=20,
       N_p=10,
       N_m=100,
       Nt_total=1000,
       epochs=5,
       adam_inner=100,
       use_lbfgs=True,
       dtype="float64",
       verbose=True,
   )


7. Run the Forward Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

The system is released from

.. code-block:: python

   x0 = [np.pi / 3, 0.0]

and solved over ten seconds.

.. code-block:: python

   result = adalib.run_forward(
       system=system,
       x0=[np.pi / 3, 0.0],
       t_span=(0.0, 10.0),
       options=options,
   )

No change to the solver interface is required even though the governing
equations differ from the Lotka–Volterra and Euler systems.


8. Inspect the Result
~~~~~~~~~~~~~~~~~~~~~

The returned result object provides direct access to the predicted state
trajectories.

.. code-block:: python

   print(
       f"t: {result.t.shape} "
       f"range [{result.t[0]:.2f}, "
       f"{result.t[-1]:.2f}] s"
   )

   print(
       f"theta: "
       f"[{result.y[0].min():.4f}, "
       f"{result.y[0].max():.4f}] rad"
   )

   print(
       f"omega: "
       f"[{result.y[1].min():.4f}, "
       f"{result.y[1].max():.4f}] rad/s"
   )


9. Visualization
~~~~~~~~~~~~~~~~

Unlike the earlier examples, the standardized ADALib result object can
directly generate a forward-solution plot using ``forward_plot``.

.. code-block:: python

   fig, axes = result.forward_plot(
       state_names=[
           r"$\theta$ [rad]",
           r"$\omega$ [rad/s]",
       ],
       save_path="pendulum_forward_result.png",
       show=False,
   )

This provides a convenient post-processing interface without requiring the
user to manually construct the visualization from ``result.t`` and
``result.y``.

.. figure:: pendulum_forward_result.png
   :width: 75%
   :align: center
   :alt: Damped nonlinear pendulum forward solution


Complete Source Code
--------------------

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
