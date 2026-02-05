import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_nonparasccs(x, type='line', conf_int=0.95, **kwargs):
    """
    Plot non-parametric SCCS results.
    
    Parameters:
    x : fitted model object
        Should have attributes: exposure, se, ageaxis, age, timesinceexpo.
    type : str, default 'line'
        Plot type ('line' for line plot).
    conf_int : float, default 0.95
        Confidence interval level.
    """
    fit = x
    z = norm.ppf((1 + conf_int) / 2)
    
    rho = fit.exposure
    lci = rho * np.exp(-z * fit.se / rho)
    uci = rho * np.exp(z * fit.se / rho)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Age plot
    axes[0].plot(fit.ageaxis, fit.age, type if type != 'line' else '-')
    axes[0].set_ylabel('Relative incidence')
    axes[0].set_xlabel('Age (days)')
    
    # Exposure plot
    axes[1].plot(fit.timesinceexpo, fit.exposure, type if type != 'line' else '-')
    axes[1].plot(fit.timesinceexpo, lci, '--')
    axes[1].plot(fit.timesinceexpo, uci, '--')
    axes[1].set_ylabel('Relative incidence')
    axes[1].set_xlabel('Days since start of risk period')
    axes[1].set_ylim(0, max(uci) + 2)
    
    plt.tight_layout()
    plt.show()