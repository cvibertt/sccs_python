ConvSCCS Python Prototype
=========================

This folder contains a lightweight Python implementation of the ConvSCCS model
described in `ConvSCCS.pdf`. The implementation targets the discrete-time SCCS
formulation with convolutional drug effects and a penalized objective:

    -loglik(φ, θ) + γ_tv * TV(θ) + γ_gl * group_lasso(θ)

Key ideas from the paper
------------------------
- Discrete-time SCCS with conditional Poisson likelihood.
- Convolutional lag effect for each drug exposure.
- Step-function effects over a risk window of length `p`.
- Penalization combining total-variation (piecewise constant lags) and
  group-lasso (drug selection).

What's implemented
------------------
- `convsccs.py`:
  - Conditional Poisson negative log-likelihood.
  - Gradient for φ and θ.
  - Proximal gradient (FISTA + backtracking).
  - Dykstra-based prox for TV + group-lasso per drug.
- `penalties.py`:
  - 1D total variation prox (Condat).
  - Group-lasso prox.

Input format (prototype)
------------------------
- `y_list`: list of arrays, one per patient, shape `(K,)`, counts per interval.
- `x_list`: list of arrays, one per patient, shape `(d, K)` with point exposures.

Example usage (sketch)
----------------------
from convsccs import ConvSCCS
from data_utils import make_patient_arrays

model = ConvSCCS(p=24, gamma_tv=0.1, gamma_gl=0.1, max_iter=200)
model.fit(y_list, x_list)
theta = model.theta_
phi = model.phi_

Data binning utilities
----------------------
The `data_utils.py` module contains helpers to bin event times and point
exposures into discrete intervals:

- `build_time_grid(start, end, bin_size)`
- `bin_events(event_times, start, end, bin_size)`
- `bin_point_exposures(exposure_times, start, end, bin_size)`
- `make_patient_arrays(...)` to build `y_list` and `x_list`

Age effects (optional)
----------------------
You can add age-group effects by passing `age_bins` to `make_patient_arrays`
and supplying `age_design_list` to `ConvSCCS.fit(...)`. When using age effects,
set `use_phi=False` to avoid identifiability issues with the baseline.

Notes
-----
- This is a prototype for experimentation; it is not optimized for scale.
- The prox for TV + group-lasso uses Dykstra iterations for the sum of penalties.
