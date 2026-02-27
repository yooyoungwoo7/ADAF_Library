Lotka–Volterra (ADAF_seq)
=========================

Problem setup
-------------

We solve the Lotka–Volterra predator–prey system on a time interval :math:`t \in [0, 1]`.
The two state variables are

- :math:`r(t)`: prey (normalized)
- :math:`p(t)`: predator (normalized)

The initial condition is given by

.. math::

   r(0)=\frac{100}{U}, \qquad p(0)=\frac{15}{U},

where :math:`U=200` and :math:`R=20` are scaling constants.


Implementation
--------------

This section walks through the implementation step-by-step. The complete runnable source code is stored at the end. 


1) Import libraries
~~~~~~~~~~~~~~~~~~~

We first import the ADAF_seq library and common numerical/plotting utilities.
SciPy ``solve_ivp`` is used only to compute a reference solution for validation.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 1-7


2) Define constants, time interval, and initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the scaling constants (:math:`U, R`), the time interval bounds (:math:`lb, ub`),
and initial conditions ``ic = [r(0), p(0)]``.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 10-18


3) Define the ODE residual function (callable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ADAF_seq solver expects a callable function ``ode_res(var_list, i)``, which returns the residual of the system of ODE. Here, ``var_list[k]`` provides a pair ``(y_k, y_k_t)`` corresponding to the state and its time derivative.
The function should return the residual for equation index ``i``:

- ``i=0``: prey equation residual
- ``i=1``: predator equation residual

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 20-30


4) Configure solver options (grid / Adam / L-BFGS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We construct three option objects before calling the solver:

- ``GridOptions``: global sampling + segmentation setup
- ``AdamOptions``: Adam training hyperparameters
- ``LBFGSOptions``: optional L-BFGS refinement stage

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 34-37


5) Run ADAF_seq solver and extract the solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver is executed through ``ADAF_seq.solve_ivp``.
The returned ``solver.solution`` can be used to verify the model output to the numerical solution:

- ``t``: time array of shape ``(Nt_total,)``
- ``y``: state array of shape ``(ode_num, Nt_total)``

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 39-47

6) Compute a numerical reference solution (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For validation, we solve the same ODE system using SciPy ``solve_ivp`` evaluated on the same grid ``t``.
We set tight tolerances to obtain a high-accuracy reference.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 50-58

7) Plot time-series comparison (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We compare ADAF_seq predictions against the numerical reference in a single time-series plot.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 60-72

The resulting plot of the numerical solution and the model prediction follows: 

.. figure:: docs/source/100seg_10x4.png
   :width: 90%
   :align: center
   :alt: Lotka–Volterra time-series comparison


