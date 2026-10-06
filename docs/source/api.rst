API Reference
=============

The public API is exposed at the top level of the ``adalib`` package
(``import adalib``).

Entry points
------------

.. autofunction:: adalib.run_forward
.. autofunction:: adalib.run_inverse
.. autofunction:: adalib.run_operator
.. autofunction:: adalib.run_mpc
.. autofunction:: adalib.data_gen

Options
-------

.. autoclass:: adalib.ForwardOptions
   :members:

.. autoclass:: adalib.InverseOptions
   :members:

.. autoclass:: adalib.InverseParameter
   :members:

.. autoclass:: adalib.OperatorOptions
   :members:

.. autoclass:: adalib.MPCOptions
   :members:

Systems
-------

.. autofunction:: adalib.get_system
.. autofunction:: adalib.list_systems
.. autofunction:: adalib.register_system

.. autoclass:: adalib.ODESystem
   :members:

.. autoclass:: adalib.CallableODESystem
   :members:

Results
-------

.. autoclass:: adalib.ForwardResult
   :members:

.. autoclass:: adalib.InverseResult
   :members:

.. autoclass:: adalib.OperatorResult
   :members:

.. autoclass:: adalib.MPCResult
   :members:

.. autoclass:: adalib.ObservationData
   :members:
