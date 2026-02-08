#!/usr/bin/env python3
"""
Full SCCS Analysis Script for apdat.csv

This script runs the complete Self-Controlled Case Series (SCCS) analysis on apdat.csv,
matching R's standardsccs output. It uses the exact Hessian for precise computations.

Requirements:
- Python 3.x
- NumPy, Pandas, Patsy, SciPy
- The SCCS modules: formatdata.py, standardsccs.py, clogit.py, etc.

Usage:
- Place apdat.csv in the same directory.
- Run: python3 run_full_sccs_analysis.py
"""

import numpy as np
import pandas as pd
from python_sccs.standardsccs import standardsccs

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

    # Define age groups (matching R's agedem <- floor(seq(70, 95, 5)*365.25))
    agedem = [int(np.floor(age * 365.25)) for age in range(70, 96, 5)]
    print(f"Age groups (agedem): {agedem}")

    # Run SCCS analysis with exact Hessian
    print("\nRunning SCCS analysis with exact Hessian...")
    print("Formula: drug_0 + age")
    print("Exposure groups: [0, 1, 2]")
    print("Washout: [1, 92, 182]")
    print("Age groups:", agedem)

    try:
        result = standardsccs(
            formula='drug_0 + age',
            indiv='case',
            astart='sta',
            aend='end',
            aevent='stro',
            adrug='ap',
            aedrug='endap',
            expogrp=[0, 1, 2],
            washout=[1, 92, 182],
            agegrp=agedem,
            data=apdat,
            dataformat='multi'
        )

        print("\nAnalysis Results:")
        print(result)

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Ensure all SCCS modules are in the 'python' directory and dependencies are installed.")

if __name__ == "__main__":
    main()