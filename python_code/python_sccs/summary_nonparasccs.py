import numpy as np
from scipy.stats import norm, chi2

def summary_nonparasccs(object, conf_int=0.95):
    """
    Create summary for non-parametric SCCS model.
    
    Parameters:
    object : fitted model object
        Should have coef, se_age, smoothingpara, cv.
    conf_int : float
        Confidence interval level.
    
    Returns:
    rval : dict-like object
        Summary statistics.
    """
    class SummaryNonParasccs:
        def __init__(self, coefficients, conf_int, nevent, smp, crossvalidation):
            self.coefficients = coefficients
            self.conf_int = conf_int
            self.nevent = nevent
            self.smp = smp
            self.crossvalidation = crossvalidation
    
    fit = object
    beta = fit.coef
    se = fit.se_age
    
    z = beta / se
    pval = 1 - chi2.cdf(z**2, 1)
    
    coefficients = np.column_stack([beta, np.exp(beta), se, z, pval])
    coefficients = pd.DataFrame(coefficients, columns=["coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"])
    
    z_ci = norm.ppf((1 + conf_int) / 2)
    conf_int = np.column_stack([
        np.exp(beta),
        np.exp(-beta),
        np.exp(beta - z_ci * se),
        np.exp(beta + z_ci * se)
    ])
    conf_int = pd.DataFrame(conf_int, columns=[
        "exp(coef)", "exp(-coef)",
        f"lower .{round(100 * conf_int, 2)}",
        f"upper .{round(100 * conf_int, 2)}"
    ])
    
    nevent = None
    smp = f"{fit.smoothingpara:.2e}"
    crossvalidation = fit.cv
    
    rval = SummaryNonParasccs(coefficients, conf_int, nevent, smp, crossvalidation)
    return rval