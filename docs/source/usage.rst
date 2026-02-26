Usage
=====

This page explains how to use **ADAF_Library** to solve systems of ordinary differential equations (ODEs).

Installation
------------

Clone the repository and install documentation/build dependencies:

.. code-block:: bash

   git clone https://github.com/yooyoungwoo7/ADAF_Library.git
   cd ADAF_Library
   pip install -r docs/requirements.txt

Library usage
-------------

The core solver implementation is provided under ``pinn_lib/``.
This documentation will be updated with minimal runnable examples (e.g., Lotka–Volterra, Euler rigid body).

Project structure (high level)
------------------------------

- ``pinn_lib/ADAF``: ADAF solver core components
- ``pinn_lib/ADAF_seq``: sequential ADAF solver workflow
- ``pinn_lib/PINN``: PINN-related utilities and models
