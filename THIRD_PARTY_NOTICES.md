# Third-party and vendored code notices

## Vendored legacy research backend

ADAlib includes a vendored legacy research backend under `adalib/_vendor/legacy/`.

This backend was originally developed as part of the ADA-based ODE solver /
Operator-MPC research project at Kyung Hee University and is distributed with
permission under the same license as ADAlib (see `LICENSE`).

### Forward backend (`adalib/_vendor/legacy/forward_problem_original/`)

Contains the forward ODE solvers for the Fourier-series (ADA-F) and
Legendre-polynomial (ADA-L / LPA) bases of the Anti-Derivative Approximator.

Original authors: ADA research group, Kyung Hee University.

### Operator / MPC backend (`adalib/_vendor/legacy/operator_mpc_original/`)

Contains the operator learning training loop, dataset builder, problem
definitions (CSTR, triple-tank, bioreactor), and the receding-horizon MPC
controller.

Original authors: ADA research group, Kyung Hee University.

---

## CSTR and triple-tank benchmark parameters

The CSTR and triple-tank system parameters used in
`adalib/_vendor/legacy/operator_mpc_original/cstr_mpc_op/problems/` are
derived from the **do-mpc** benchmark suite:

> Lucia, S., Tatulea-Codrean, A., Schoppmeyer, C., & Engell, S. (2017).
> Rapid development of modular and sustainable nonlinear model predictive
> control solutions. *Control Engineering Practice*, 60, 51–62.
> https://doi.org/10.1016/j.conengprac.2016.12.009

do-mpc is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
The parameters are numerical constants (reaction rate constants, heat transfer
coefficients, etc.) taken from the published benchmark description.  Only the
parameter values — not the do-mpc source code — are used in this project.

---

## Other third-party dependencies

Runtime dependencies (numpy, scipy, sympy, tensorflow, matplotlib, tqdm, h5py)
are declared in `pyproject.toml` and are not bundled in this package.
Each is governed by its own license.
