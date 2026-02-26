Lotka_Volterra (ADAF_seq)
=====================

Problem setup

We solve the Lotka–Volterra predator–prey system on a normalized time domain 
𝑡
∈
[
0
,
1
]
t∈[0,1].
The two state variables are:

𝑟
(
𝑡
)
r(t): prey population (normalized by 
𝑈
U)

𝑝
(
𝑡
)
p(t): predator population (normalized by 
𝑈
U)

Initial conditions are given as:

𝑟
(
0
)
=
100
𝑈
,
𝑝
(
0
)
=
15
𝑈
,
r(0)=
U
100
	​

,p(0)=
U
15
	​

,

with scaling constants 
𝑈
=
200
U=200, 
𝑅
=
20
R=20.

Implementation

This section walks through the full implementation of an ADAF_seq solver run for the Lotka–Volterra system, including (i) residual definition, (ii) solver configuration, (iii) solver execution, and (iv) comparison with a numerical reference solution obtained via SciPy.

1) Import libraries

We first import the ADAF_seq API from pinn_lib, along with common scientific Python utilities.

import pinn_lib
from pinn_lib import ADAF_seq

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ADAF_seq provides the sequential training-based ODE solver interface.

solve_ivp is used only as a reference numerical integrator for validation.

2) Define constants, domain, and initial conditions

We define scaling constants, time interval bounds, and normalized initial conditions.

U = 200.0
R = 20.0

lb = 0.0
ub = 1.0

ic = [100.0 / U,   # r(0): prey
      15.0  / U]   # p(0): predator

lb, ub specify the solution interval.

ic is a list of initial values for the ODE state vector 
[
𝑟
(
0
)
,
𝑝
(
0
)
]
[r(0),p(0)].

3) Define the ODE residual function (callable)

The ADAF_seq solver expects a residual function that evaluates the ODE constraint for each equation index i.
Here, var_list[k] contains a pair 
(
𝑦
𝑘
,
𝑑
𝑦
𝑘
𝑑
𝑡
)
(y
k
	​

,
dt
dy
k
	​

	​

).

def test3(var_list, i):
    r, r_t = var_list[0]
    p, p_t = var_list[1]

    if i == 0:
        rhs = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
        return r_t - rhs

    elif i == 1:
        rhs = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)
        return p_t - rhs

Interpretation:

Equation 1 (prey): 
𝑑
𝑟
𝑑
𝑡
−
𝑓
𝑟
(
𝑟
,
𝑝
)
=
0
dt
dr
	​

−f
r
	​

(r,p)=0

Equation 2 (predator): 
𝑑
𝑝
𝑑
𝑡
−
𝑓
𝑝
(
𝑟
,
𝑝
)
=
0
dt
dp
	​

−f
p
	​

(r,p)=0

The solver minimizes these residuals over the time grid, subject to initial conditions.

4) Configure solver options (grid / optimizer)

We build three option objects:

GridOptions: discretization and segmentation setup

AdamOptions: training setup for Adam optimizer

LBFGSOptions: whether to apply L-BFGS refinement

grid  = ADAF_seq.GridOptions(lb=lb, ub=ub, Nt_total=2000, n_seg=10, Nt_seg=None, gamma=0.8, L=1.0)
adam  = ADAF_seq.AdamOptions(epochs=20, inner=50, lr=1e-3, seed=0, dtype="float64", xla_predict=True, xla_step=False)
lbfgs = ADAF_seq.LBFGSOptions(use=True)

Notes:

Nt_total=2000 defines total evaluation points across the full time domain.

n_seg=10 partitions the domain into 10 segments for sequential training.

dtype="float64" enforces higher precision (often important for stability/accuracy).

xla_predict=True enables XLA compilation for inference (speed-up).

5) Run the ADAF_seq solver

We now call the library API. The solver returns an object containing a solution field compatible with SciPy’s solve_ivp output format.

solver = ADAF_seq.solve_ivp(ode_res=test3, ic=ic, grid=grid, adam=adam, lbfgs=lbfgs, verbose=True)

t = solver.solution.t                 # (Nt_total,)
y = solver.solution.y                 # (ode_num, Nt_total)

r_pred = y[0]
p_pred = y[1]

solver.solution.t is the time array used for plotting.

solver.solution.y stores the predicted solution 
[
𝑟
(
𝑡
)
,
𝑝
(
𝑡
)
]
[r(t),p(t)].

6) Compute a numerical reference solution (SciPy)

To validate the ADAF_seq output, we solve the same system using SciPy’s explicit ODE integrator.

def rhs(t, y):
    r, p = y
    dr = (R / U) * (2.0 * U * r - 0.04 * (U ** 2) * r * p)
    dp = (R / U) * (0.02 * (U ** 2) * r * p - 1.06 * U * p)
    return [dr, dp]

sol = solve_ivp(rhs, (lb, ub), [ic[0], ic[1]], t_eval=t, rtol=1e-10, atol=1e-12)
r_num = sol.y[0]
p_num = sol.y[1]

t_eval=t ensures both solutions are evaluated at identical time points.

Tight tolerances are used to obtain a high-accuracy reference.

7) Plot and compare (time series)

Finally, we plot the ADAF_seq solution and the numerical reference together.

plt.figure(figsize=(10, 4))
plt.plot(t, r_pred,  label="r ADAF_seq")
plt.plot(t, r_num, "--", label="r numerical")
plt.plot(t, p_pred,  label="p ADAF_seq")
plt.plot(t, p_num, "--", label="p numerical")
plt.xlabel("t")
plt.ylabel("states")
plt.title("Lotka–Volterra: time series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

This comparison provides a direct visual check of whether the sequential ADAF solver reproduces the expected prey–predator dynamics over the full time interval.
