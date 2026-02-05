#!/usr/bin/env python3
"""
Test script for exact Hessian in SCCS analysis.
"""

import numpy as np
import pandas as pd
from python.simulatesccsdata import simulatesccsdata
from python.formatdata import formatdata
from python.standardsccs import standardsccs

# Simulate data
print("Simulating SCCS data...")
data = simulatesccsdata(
    nindivs=500,
    astart=0,
    aend=1,
    adrug=[0.2, 0.6],  # Exposure start times for two exposures
    aedrug=[0.4, 0.8],  # Exposure end times
    eexpo=[2.0, 1.5],  # Relative incidences
    expogrp=[1, 2],
    agegrp=[0, 0.3, 0.7],
    eage=[1.0, 1.2, 1.1]
)

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print("Sample data:")
print(data.head())

# Run formatdata
print("\nFormatting data...")
chopdat = formatdata(
    indiv=data['indiv'],
    astart=data['astart'],
    aend=data['aend'],
    aevent=data['aevent'],
    adrug=data['adrug'],
    aedrug=data['aedrug'],
    expogrp=[1, 2],
    agegrp=[0, 0.3, 0.7],
    dataformat='multi'
)

print("Chopped data shape:", chopdat.shape)
print("Chopped columns:", chopdat.columns.tolist())
print("Sample chopped data:")
print(chopdat.head())

# Run standardsccs with exact Hessian
print("\nRunning SCCS analysis with exact Hessian...")
result = standardsccs(
    formula='drug_0 + age',
    indiv='indivL',
    astart=None,
    aend=None,
    aevent=None,
    adrug=None,
    aedrug=None,
    expogrp=None,
    agegrp=None,
    data=chopdat,
    dataformat='stack'
)

print("\nResults:")
print(f"Coefficients: {result.params}")
print(f"Exp(Coefficients): {np.exp(result.params)}")
print(f"Standard Errors: {result.se}")
print(f"Wald Statistics: {result.wald_stats}")
print(f"Wald p-values: {result.p_wald}")
print(f"Overall Wald p-value: {result.p_overall_wald}")
print(f"Log-likelihood: {result.llf}")
print(f"Concordance: {result.concordance}")
print(f"LR Statistic: {result.lr_stat}")
print(f"LR p-value: {result.p_lr}")

print("\nAnalysis complete!")