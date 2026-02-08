#!/usr/bin/env python3
"""
Nonparametric SCCS Analysis Script for apdat.csv

This script runs the nonparametric Self-Controlled Case Series (SCCS) analysis on apdat.csv,
using smooth spline-based effects for age and exposure (M-splines by default).

Requirements:
- Python 3.x
- NumPy, Pandas, Patsy, SciPy
- The SCCS modules: nonparasccs.py, formatdata.py, etc.

Usage:
- Place apdat.csv in the same directory.
- Run: python3 run_nonparametric_sccs.py
"""

import numpy as np
import pandas as pd
from python_sccs.nonparasccs import nonparasccs

def main():
    print("Loading apdat.csv...")
    try:
        apdat = pd.read_csv('apdat.csv')
    except FileNotFoundError:
        print("Error: apdat.csv not found. Please place it in the current directory.")
        return

    print(f"Data shape: {apdat.shape}")
    print("Sample data:")
    print(apdat.head())

    # Run nonparametric SCCS analysis
    print("\nRunning nonparametric SCCS analysis...")
    print("Spline type: dmspline (internal)")
    print("Knots: 6 for age, 6 for exposure")

    try:
        result = nonparasccs(
            indiv=apdat['case'].values,
            astart=apdat['sta'].values,
            aend=apdat['end'].values,
            aevent=apdat['stro'].values,
            adrug=apdat['ap'].values,
            aedrug=apdat['endap'].values,
            kn1=6,
            kn2=6,
            data=apdat
        )

        print("\nNonparametric SCCS Results:")
        print(f"Converged: {getattr(result, 'success', 'Unknown')}")
        print(f"Log-likelihood: {getattr(result, 'fun', 'N/A')}")
        print(f"Parameters: {getattr(result, 'x', 'N/A')}")

        # Run print.nonparasccs
        print("\nRunning print.nonparasccs...")
        from python_sccs.print_nonparasccs import print_nonparasccs
        print_nonparasccs(result)

        # Run plot.nonparasccs (assuming it saves or displays plots)
        print("\nRunning plot.nonparasccs...")
        from python_sccs.plot_nonparasccs import plot_nonparasccs
        plot_nonparasccs(result)
        print("Plots generated (check for saved files or display).")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Ensure all SCCS modules are in the 'python' directory and dependencies are installed.")

if __name__ == "__main__":
    main()