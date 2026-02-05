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
from python.nonparasccs import nonparasccs

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

    # Define age groups for chopping (optional, but helps with data format)
    agedem = [int(np.floor(age * 365.25)) for age in range(70, 96, 5)]
    print(f"Age groups (agedem): {agedem}")

    # Run nonparametric SCCS analysis
    print("\nRunning nonparametric SCCS analysis...")
    print("Formula: event ~ exposure + age (smooth effects)")
    print("Spline type: mspline (M-splines)")
    print("Spline DF: 5 for age, 5 for exposure")
    print("Max age: 1.5, Min age: 0.0")
    print("Max exp: 1.5, Min exp: 0.0")

    try:
        result = nonparasccs(
            indiv='case',
            astart='sta',
            aend='end',
            aevent='stro',
            adrug='ap',
            aedrug='endap',
            expogrp=[0, 1, 2],  # Example exposure groups
            washout=[1, 92, 182],
            agegrp=agedem,
            data=apdat,
            dataformat='multi',
            spline_df={'age': 5, 'exposure': 5},  # Degrees of freedom for splines
            spline_type='mspline',  # M-splines
            max_age=1.5,  # Max age in years (adjust based on data)
            min_age=0.0,
            max_exp=1.5,  # Max exposure time
            min_exp=0.0,
            niter=50,  # Max iterations
            step=0.1   # Step size for optimization
        )

        print("\nNonparametric SCCS Results:")
        print(f"Converged: {result.get('converged', 'Unknown')}")
        print(f"Log-likelihood: {result.get('ll', 'N/A')}")
        print(f"Parameters: {result.get('params', 'N/A')}")
        print(f"Age spline coefficients: {result.get('age_spline', 'N/A')}")
        print(f"Exposure spline coefficients: {result.get('exp_spline', 'N/A')}")

        # Run print.nonparasccs
        print("\nRunning print.nonparasccs...")
        from python.print_nonparasccs import print_nonparasccs
        print_nonparasccs(result)

        # Run plot.nonparasccs (assuming it saves or displays plots)
        print("\nRunning plot.nonparasccs...")
        from python.plot_nonparasccs import plot_nonparasccs
        plot_nonparasccs(result)
        print("Plots generated (check for saved files or display).")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Ensure all SCCS modules are in the 'python' directory and dependencies are installed.")

if __name__ == "__main__":
    main()