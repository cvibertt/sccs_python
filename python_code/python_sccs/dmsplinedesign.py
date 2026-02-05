import numpy as np
from scipy.interpolate import BSpline

def dmsplinedesign(x, knots1, m, deriv=0):
    """
    M-spline functions and their Derivatives.
    
    Parameters:
    x : array-like
        Event times or values to evaluate the splines at.
    knots1 : array-like
        The internal knots.
    m : int
        The order of the M-spline (degree + 1).
    deriv : int
        The order of the derivative (default 0).
    
    Returns:
    design : ndarray
        The design matrix of M-splines (or their derivatives).
    """
    x = np.asarray(x)
    knots1 = np.asarray(knots1)
    
    # Assuming uniform spacing for extension
    if len(knots1) > 1:
        dk = knots1[1] - knots1[0]
    else:
        dk = 1  # arbitrary if only one knot
    
    # Extend knots: add m-1 knots on each side
    left_extend = knots1[0] - dk * np.arange(m-1, 0, -1)
    right_extend = knots1[-1] + dk * np.arange(1, m)
    knots = np.concatenate([left_extend, knots1, right_extend])
    
    n_basis = len(knots) - m
    design = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Create coefficient vector for the i-th basis function
        c = np.zeros(n_basis)
        c[i] = 1
        # BSpline in scipy: k is degree, so degree = m-1
        bs = BSpline(knots, c, k=m-1)
        # Evaluate the derivative at x
        design[:, i] = bs(x, nu=deriv) * (m / (knots[i + m] - knots[i]))
    
    return design