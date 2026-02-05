import numpy as np
from .dmsplinedesign import dmsplinedesign

def ispline(x, knots1, m):
    """
    Compute integral of M-splines to give I-splines.
    
    Parameters:
    x : array-like
        Values to compute the I-spline at.
    knots1 : array-like
        Knots.
    m : int
        Order.
    
    Returns:
    resu : ndarray
        I-spline matrix.
    """
    x = np.asarray(x)
    knots1 = np.asarray(knots1)
    
    dk = knots1[1] - knots1[0]
    k = np.concatenate([
        knots1[0] - dk * np.arange(4, 0, -1),
        knots1,
        knots1[-1] + dk * np.arange(1, 5)
    ])
    
    msplinedesign5 = dmsplinedesign(x, knots1, 5, deriv=0)
    d = len(k) - m - 1
    resu = np.zeros((len(x), d))
    
    for j in range(len(x)):
        for i in range(d):
            if x[j] > k[i + m + 1]:
                resu[j, i] = 1
            elif x[j] < k[i + 1]:
                resu[j, i] = 0
            elif k[i + 1] < x[j] <= k[i + 2]:
                resu[j, i] = (k[i + m + 2] - k[i + 1]) * msplinedesign5[j, i + 1] / (m + 1)
            elif k[i + 2] < x[j] <= k[i + 3]:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) * msplinedesign5[j, i + 1] +
                              (k[i + m + 3] - k[i + 2]) * msplinedesign5[j, i + 2]) / (m + 1)
            elif k[i + 3] < x[j] <= k[i + 4]:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) * msplinedesign5[j, i + 1] +
                              (k[i + m + 3] - k[i + 2]) * msplinedesign5[j, i + 2] +
                              (k[i + m + 4] - k[i + 3]) * msplinedesign5[j, i + 3]) / (m + 1)
            else:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) * msplinedesign5[j, i + 1] +
                              (k[i + m + 3] - k[i + 2]) * msplinedesign5[j, i + 2] +
                              (k[i + m + 4] - k[i + 3]) * msplinedesign5[j, i + 3] +
                              (k[i + m + 5] - k[i + 4]) * msplinedesign5[j, i + 4]) / (m + 1)
    
    return resu[:, :d-1]