#!/usr/bin/env python3
"""
Run SCCS analysis on apdat.csv with exact Hessian, matching R's call.
"""

import numpy as np
import pandas as pd
from python.standardsccs import standardsccs

# Load data
print("Loading apdat.csv...")
apdat = pd.read_csv('apdat.csv')
print("Data shape:", apdat.shape)
print("Columns:", apdat.columns.tolist())
print("Sample data:")
print(apdat.head())

# Define age groups (matching R's agedem <- floor(seq(70, 95, 5)*365.25))
agedem = [int(np.floor(age * 365.25)) for age in range(70, 96, 5)]
print("Age groups (agedem):", agedem)

# Run standardsccs matching R's call
print("\nRunning SCCS analysis with exact Hessian...")
result = standardsccs(
    formula=' ~ ap1 + ap2 + ap3 + age',
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

print("\nResults:")
print(result)

print("\nAnalysis complete!")