import numpy as np
from scipy.interpolate import BSpline

def msplinedesign(x, k, m=4):
    """
    Compute design matrix for M-splines of order m.
    
    Parameters:
    x : array-like
        Event times.
    k : array-like
        Knots.
    m : int, default 4
        Order (degree + 1).
    
    Returns:
    design : ndarray
        M-spline design matrix.
    """
    x = np.asarray(x)
    k = np.asarray(k)
    
    n_basis = len(k) - m
    design = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1
        bs = BSpline(k, c, k=m-1)
        design[:, i] = bs(x) * m / (k[i + m] - k[i])
    
    return design