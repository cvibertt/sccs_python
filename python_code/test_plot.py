import pandas as pd
import numpy as np
from python_sccs import plot_nonparasccs
import matplotlib.pyplot as plt

# Create dummy fitted object for nonparasccs
class DummyFit:
    def __init__(self):
        self.exposure = np.exp(np.linspace(-1, 1, 50))  # Dummy exposure RR
        self.se = np.random.uniform(0.1, 0.5, 50)  # Dummy SE
        self.ageaxis = np.linspace(70, 90, 50)  # Age axis
        self.age = np.exp(np.sin(np.linspace(0, 2*np.pi, 50)))  # Dummy age RR
        self.timesinceexpo = np.linspace(0, 30, 50)  # Time since exposure

fit = DummyFit()
plot_nonparasccs(fit)

# Save the plot
plt.savefig('sccs_plot.png')
print("Plot saved as 'sccs_plot.png'")