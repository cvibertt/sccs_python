import numpy as np
import pandas as pd
from scipy.stats import chi2

def summary_sccs(mod, sandwich=None, ses=None, ncases=None, nevents=None):
    """
    Create summary for SCCS model.
    
    Parameters:
    mod : fitted model (statsmodels GLM result)
    sandwich : variance-covariance matrix
    ses : standard errors
    ncases : number of cases
    nevents : number of events
    
    Returns:
    rval : dict-like object with summary stats
    """
    class SummarySCCS:
        def __init__(self, coefficients, conf_int, VarCov, n, nevent):
            self.coefficients = coefficients
            self.conf_int = conf_int
            self.VarCov = VarCov
            self.n = n
            self.nevent = nevent
    
    coef = mod.params
    if ses is None:
        ses = np.sqrt(np.diag(mod.cov_params()))
    
    z = coef / ses
    pval = 1 - chi2.cdf(z**2, 1)
    
    coefficients = pd.DataFrame({
        'coef': coef,
        'exp(coef)': np.exp(coef),
        'se(coef)': ses,
        'z': z,
        'Pr(>|z|)': pval
    })
    
    exp_coef = np.exp(coef)
    lower = np.exp(coef - 1.96 * ses)
    upper = np.exp(coef + 1.96 * ses)
    exp_neg_coef = np.exp(-coef)
    
    conf_int = pd.DataFrame({
        'exp(coef)': exp_coef,
        'exp(-coef)': exp_neg_coef,
        'lower .95': lower,
        'upper .95': upper
    })
    
    n = f"{ncases} cases" if ncases else None
    nevent = nevents
    
    rval = SummarySCCS(coefficients, conf_int, sandwich, n, nevent)
    return rval