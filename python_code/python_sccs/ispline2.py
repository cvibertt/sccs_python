import numpy as np
from .dmsplinedesign import dmsplinedesign

def ispline2(x, knots1, m):
    """
    Evaluate second integral of an I-spline.
    """
    x = np.asarray(x)
    knots1 = np.asarray(knots1)
    
    dk = knots1[1] - knots1[0]
    k = np.concatenate([
        knots1[0] - dk * np.arange(4, 0, -1),
        knots1,
        knots1[-1] + dk * np.arange(1, 5)
    ])
    k7 = np.concatenate([
        knots1[0] - dk * np.arange(6, 0, -1),
        knots1,
        knots1[-1] + dk * np.arange(1, 7)
    ])
    
    msplinedesign7 = dmsplinedesign(x, knots1, 7, deriv=0)
    
    d = len(k) - m - 1
    resu = np.zeros((len(x), d))
    
    for j in range(len(x)):
        for i in range(d):
            if x[j] > k[i + m + 1]:
                resu[j, i] = (x[j] - k[i + m + 1]) + \
                    ((k[i + m + 2] - k[i + 1]) / (m + 1)) * \
                    (((k7[i + m + 3] - k7[i + 1]) / (m + 3)) * msplinedesign7[j, i + 2] + \
                     ((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5] + \
                     ((k7[i + m + 7] - k7[i + 5]) / (m + 3)) * msplinedesign7[j, i + 6]) + \
                    ((k[i + m + 3] - k[i + 2]) / (m + 1)) * \
                    (((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5] + \
                     ((k7[i + m + 7] - k7[i + 5]) / (m + 3)) * msplinedesign7[j, i + 6]) + \
                    ((k[i + m + 4] - k[i + 3]) / (m + 1)) * \
                    (((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5] + \
                     ((k7[i + m + 7] - k7[i + 5]) / (m + 3)) * msplinedesign7[j, i + 6]) + \
                    ((k[i + m + 5] - k[i + 4]) / (m + 1)) * \
                    (((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5] + \
                     ((k7[i + m + 7] - k7[i + 5]) / (m + 3)) * msplinedesign7[j, i + 6])
            elif x[j] < k[i + 1]:
                resu[j, i] = 0
            elif k[i + 1] < x[j] <= k[i + 2]:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) / (m + 1)) * ((k7[i + m + 3] - k7[i + 1]) / (m + 3)) * msplinedesign7[j, i + 2]
            elif k[i + 2] < x[j] <= k[i + 3]:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) / (m + 1)) * \
                    (((k7[i + m + 3] - k7[i + 1]) / (m + 3)) * msplinedesign7[j, i + 2] + \
                     ((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3]) + \
                    ((k[i + m + 3] - k[i + 2]) / (m + 1)) * \
                    (((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3])
            elif k[i + 3] < x[j] <= k[i + 4]:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) / (m + 1)) * \
                    (((k7[i + m + 3] - k7[i + 1]) / (m + 3)) * msplinedesign7[j, i + 2] + \
                     ((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4]) + \
                    ((k[i + m + 3] - k[i + 2]) / (m + 1)) * \
                    (((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4]) + \
                    ((k[i + m + 4] - k[i + 3]) / (m + 1)) * \
                    (((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4])
            else:
                resu[j, i] = ((k[i + m + 2] - k[i + 1]) / (m + 1)) * \
                    (((k7[i + m + 3] - k7[i + 1]) / (m + 3)) * msplinedesign7[j, i + 2] + \
                     ((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5]) + \
                    ((k[i + m + 3] - k[i + 2]) / (m + 1)) * \
                    (((k7[i + m + 4] - k7[i + 2]) / (m + 3)) * msplinedesign7[j, i + 3] + \
                     ((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5]) + \
                    ((k[i + m + 4] - k[i + 3]) / (m + 1)) * \
                    (((k7[i + m + 5] - k7[i + 3]) / (m + 3)) * msplinedesign7[j, i + 4] + \
                     ((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5]) + \
                    ((k[i + m + 5] - k[i + 4]) / (m + 1)) * \
                    (((k7[i + m + 6] - k7[i + 4]) / (m + 3)) * msplinedesign7[j, i + 5])
    
    return resu[:, :d-1]