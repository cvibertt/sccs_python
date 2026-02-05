import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

def summary_smoothagesccs(object, conf_int=0.95):
    """
    Create summary for smoothed age SCCS model.
    
    Parameters:
    object : fitted model object
        Should have coef, se, smp, smoothingpara, cv.
    conf_int : float
        Confidence interval level.
    
    Returns:
    rval : dict-like object
        Summary statistics.
    """
    class SummarySmoothAgesccs:
        def __init__(self, coefficients, conf_int, nevent, smp, smoothingpara, cv):
            self.coefficients = coefficients
            self.conf_int = conf_int
            self.nevent = nevent
            self.smp = smp
            self.smoothingpara = smoothingpara
            self.cv = cv
    
    fit = object
    beta = np.array(fit.coef)
    se = np.array(fit.se)
    
    z = beta / se
    pval = 1 - chi2.cdf(z**2, 1)
    
    coefficients = pd.DataFrame({
        "coef": beta,
        "exp(coef)": np.exp(beta),
        "se(coef)": se,
        "z": z,
        "Pr(>|z|)": pval
    })
    
    z_ci = norm.ppf((1 + conf_int) / 2)
    conf_int = pd.DataFrame({
        "exp(coef)": np.exp(beta),
        "exp(-coef)": np.exp(-beta),
        f"lower .{round(100 * conf_int, 2)}": np.exp(beta - z_ci * se),
        f"upper .{round(100 * conf_int, 2)}": np.exp(beta + z_ci * se)
    })
    
    nevent = None
    smp = fit.smp
    smoothingpara = f"{fit.smoothingpara:.2e}"
    cv = fit.cv
    
    rval = SummarySmoothAgesccs(coefficients, conf_int, nevent, smp, smoothingpara, cv)
    return rval