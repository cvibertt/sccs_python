import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_smoothexposccs(x, type='line', conf_int=0.95, **kwargs):
    """
    Plot smoothed exposure effect for SCCS.
    
    Parameters:
    x : fitted model object
        Should have 'exposure', 'se', 'timesinceexpo'.
    type : str, default 'line'
        Plot type.
    conf_int : float, default 0.95
        Confidence interval.
    """
    fit = x
    z = norm.ppf((1 + conf_int) / 2)
    
    rho = fit.exposure
    lci = rho * np.exp(-z * fit.se / rho)
    uci = rho * np.exp(z * fit.se / rho)
    
    plt.plot(fit.timesinceexpo, rho, type if type != 'line' else '-')
    plt.plot(fit.timesinceexpo, lci, '--')
    plt.plot(fit.timesinceexpo, uci, '--')
    plt.ylabel('Relative incidence')
    plt.xlabel('Days since start of risk period')
    plt.ylim(0, max(uci) + 2)
    plt.show()