import numpy as np
from scipy.stats import norm

def samplesize(eexpo, risk, astart, aend, p, alpha=0.05, power=0.8, eage=None, agegrp=None):
    """
    Calculate sample size for SCCS study.
    
    Parameters:
    eexpo : float
        Relative incidence during exposure.
    risk : float
        Length of risk period.
    astart : float
        Start age.
    aend : float
        End age.
    p : array-like
        Probabilities for age groups.
    alpha : float, default 0.05
        Significance level.
    power : float, default 0.8
        Power.
    eage : array-like, optional
        Age-related relative incidences.
    agegrp : array-like, optional
        Age group cut points.
    
    Returns:
    n : int
        Required sample size.
    """
    if agegrp is None and eage is not None:
        raise ValueError("Please specify age group cut points 'agegrp'")
    if agegrp is not None and eage is None:
        raise ValueError("Please specify age related relative incidences 'eage'")
    
    if eage is not None and agegrp is not None:
        if len(p) != len(eage) + 1 or len(p) != len(agegrp) + 1:
            raise ValueError("Please specify appropriate number of eage and agegrp")
    
    if np.sum(p) <= 0 or np.sum(p) > 1:
        raise ValueError("The sum of the vector/scalar 'p' must be greater than 0 and less than or equal to 1 (0, 1]")
    
    if eage is not None:
        eage = np.concatenate([[1], eage])
    
    rho = eexpo
    if agegrp is not None:
        agecupts = np.concatenate([[astart], agegrp, [aend]])
        agegrplengths = np.diff(agecupts)
        agegrplengths[-1] += 1
        es = agegrplengths
    else:
        es = np.array([aend - astart])
    
    if risk >= np.min(es):
        raise ValueError("risk length must be less than the length of the shortest age group")
    
    estr = risk
    sss = np.sum(eage * es)
    r = np.array([(eage[i] * estr) / sss for i in range(len(es))])
    
    pi = (r * rho) / (r * rho + 1 - r)
    
    vjden = (1 - np.sum(p)) + np.sum(p * (r * rho + 1 - r))
    vj = np.array([p[i] * (r[i] * rho + 1 - r[i]) / vjden for i in range(len(es))])
    
    A = 2 * np.sum(vj * (pi * np.log(rho) - np.log(r * rho + 1 - r)))
    B = (np.log(rho) ** 2 / A) * np.sum(vj * pi * (1 - pi))
    
    za = round(norm.ppf(1 - alpha / 2), 4)
    zb = round(norm.ppf(power), 4)
    
    n = ((za + zb * np.sqrt(B)) ** 2) / A
    return int(np.ceil(n))