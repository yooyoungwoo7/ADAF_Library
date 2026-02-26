
Lotka–Volterra (ADAF_seq)
=========================

Problem setup
-------------

We solve a normalized Lotka–Volterra predator–prey system on a time interval :math:`t \in [0, 1]`.
The two state variables are

- :math:`r(t)`: prey (normalized)
- :math:`p(t)`: predator (normalized)

The initial condition is given by

.. math::

   r(0)=\frac{100}{U}, \qquad p(0)=\frac{15}{U},

where :math:`U=200` and :math:`R=20` are scaling constants used in the residual definition.

Implementation
--------------

This section walks through the implementation step-by-step. The complete runnable script is stored in
``examples/tests_ADAF_seq_lotka.py`` and is included here using ``literalinclude``.

1) Import libraries
~~~~~~~~~~~~~~~~~~~

We first import the ADAF_seq API and common numerical/plotting utilities.
SciPy ``solve_ivp`` is used only to compute a reference solution for validation.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 1-7

2) Define constants, time interval, and initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the scaling constants (:math:`U, R`), the time interval bounds (:math:`lb, ub`),
and normalized initial conditions ``ic = [r(0), p(0)]``.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 10-18

3) Define the ODE residual function (callable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ADAF_seq solver expects a residual callback ``ode_res(var_list, i)``.
Here, ``var_list[k]`` provides a pair ``(y_k, y_k_t)`` corresponding to the state and its time derivative.
We return the residual for equation index ``i``:

- ``i=0``: prey equation residual
- ``i=1``: predator equation residual

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 20-30

4) Configure solver options (grid / Adam / L-BFGS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We construct three option objects:

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
The returned ``solver.solution`` follows SciPy ``solve_ivp`` style:

- ``t``: time array of shape ``(Nt_total,)``
- ``y``: state array of shape ``(ode_num, Nt_total)``

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 39-47

6) Compute a numerical reference solution (SciPy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For validation, we solve the same ODE system using SciPy ``solve_ivp`` evaluated on the same grid ``t``.
We set tight tolerances to obtain a high-accuracy reference.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 50-58

7) Plot time-series comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We compare ADAF_seq predictions against the numerical reference in a single time-series plot.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 60-72

8) Report relative L2 errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we compute relative L2 errors for each state and for the concatenated state vector.

.. literalinclude:: ../../examples/tests_ADAF_seq_lotka.py
   :language: python
   :linenos:
   :lines: 74-90
