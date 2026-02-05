import matplotlib.pyplot as plt

def plot_smoothagesccs(x, type='line', **kwargs):
    """
    Plot smoothed age effect for SCCS.
    
    Parameters:
    x : fitted model object
        Should have 'ageaxis' and 'age' attributes.
    type : str, default 'line'
        Plot type.
    """
    fit = x
    plt.plot(fit.ageaxis, fit.age, type if type != 'line' else '-')
    plt.ylabel('Relative incidence')
    plt.xlabel('Age (days)')
    plt.show()