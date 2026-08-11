Damped Nonlinear Pendulum
=========================

Problem Setup
-------------

This example demonstrates how a user-defined ODE system can be solved
using ADALib.

Unlike the predefined Lotka–Volterra and Euler examples, the pendulum
system is explicitly defined by the user through ``CallableODESystem``.

The governing equation is

.. math::

   \frac{d^2\theta}{dt^2}
   +
   \gamma\frac{d\theta}{dt}
   +
   \frac{g}{L}\sin\theta
   =
   0.

Introducing

.. math::

   \omega = \frac{d\theta}{dt},

the second-order equation is written as the first-order system

.. math::

   \frac{d\theta}{dt} = \omega,

.. math::

   \frac{d\omega}{dt}
   =
   -\gamma\omega
   -
   \frac{g}{L}\sin\theta.

The parameters are

.. math::

   \gamma = 0.30\ {\rm s}^{-1},
   \qquad
   \frac{g}{L}=9.81\ {\rm s}^{-2}.

The pendulum is released from

.. math::

   \theta(0)=\frac{\pi}{3},
   \qquad
   \omega(0)=0,

and solved over

.. math::

   t\in[0,10]\ {\rm s}.


Implementation
--------------

1. Define the Numerical RHS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first callable represents the conventional right-hand side of the ODE
system.

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 23-27

The function receives

``t``
   Current time.

``state``
   Current state vector.

``u``
   Optional external or control input.

``p``
   Optional system parameters.


2. Define the Physics Residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TensorFlow-compatible residual function is used during physics-informed
optimization.

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 29-36

For each state, ``var_list`` provides the state and its derivative.

For example,

.. code-block:: python

   theta, theta_t = var_list[0]
   omega, omega_t = var_list[1]

The residuals are therefore

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


3. Construct CallableODESystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The numerical and physics-informed representations are combined into a
single ADALib system object.

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 38-43

This is the principal interface for implementing a custom dynamical system
in ADALib.


4. Configure ForwardOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 46-58


5. Run the Forward Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the system and options have been defined, the custom ODE system is
solved using exactly the same high-level interface as the predefined
systems.

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 61-66

Therefore, the solver call remains

.. code-block:: python

   result = adalib.run_forward(
       system=system,
       x0=[np.pi / 3, 0.0],
       t_span=(0.0, 10.0),
       options=options,
   )


6. Inspect the Result
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 69-73


7. Visualization
~~~~~~~~~~~~~~~~

The standardized ``ForwardResult`` object provides the
``forward_plot`` method for direct visualization.

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
   :lines: 76-81

This avoids requiring users to manually reconstruct plots from the raw
solution arrays.

.. figure:: pendulum_forward_result.png
   :width: 75%
   :align: center
   :alt: Damped pendulum forward solution


Complete Source Code
--------------------

.. literalinclude:: ../../examples/test_adalib_forward_pendulum.py
   :language: python
   :linenos:
