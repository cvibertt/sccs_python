#!/usr/bin/env python3
"""
Manual Implementation of Exact Hessian for Conditional Logistic Regression

This script provides a self-contained implementation of the exact Hessian computation
for conditional logistic regression, ported from R's survival C code (coxexact).
It computes the score vector and Hessian matrix exactly for strata with one event
per stratum (standard SCCS).

Usage:
- Define your design matrix X, outcome y, strata, and offset.
- Call exact_score_hessian(X, y, strata, offset) to get u and imat.
- Use u and imat for Newton-Raphson optimization or Wald tests.

Author: AI Assistant
"""

import numpy as np

NOTDONE = -1.1  # Sentinel for uncomputed values in recursive functions

def coxd0(d, n, score, dmat, dmax):
    """
    Compute the exact denominator for conditional likelihood (recursive).

    Parameters:
    d : int - Number of deaths (tied at time).
    n : int - Number at risk.
    score : array - Exponential terms (exp(zbeta)).
    dmat : array - Memoization array (flattened, size n*dmax).
    dmax : int - Max deaths in stratum.

    Returns:
    float - Partial sum.
    """
    if d == 0:
        return 1.0
    idx = (n - 1) * dmax + d - 1
    if dmat[idx] != NOTDONE:
        return dmat[idx]
    val = score[n - 1] * coxd0(d - 1, n - 1, score, dmat, dmax)
    if d < n:
        val += coxd0(d, n - 1, score, dmat, dmax)
    dmat[idx] = val
    return val

def coxd1(d, n, score, dmat, d1, covar, dmax):
    """
    First derivative w.r.t. a covariate.

    Parameters:
    d, n, score, dmat, dmax : as above
    d1 : array - Memoization for first deriv (size n*dmax).
    covar : array - Covariate values for stratum.

    Returns:
    float
    """
    idx = (n - 1) * dmax + d - 1
    if d1[idx] != NOTDONE:
        return d1[idx]
    val = score[n - 1] * covar[n - 1] * coxd0(d - 1, n - 1, score, dmat, dmax)
    if d < n:
        val += coxd1(d, n - 1, score, dmat, d1, covar, dmax)
    if d > 1:
        val += score[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1, covar, dmax)
    d1[idx] = val
    return val

def coxd2(d, n, score, dmat, d1j, d1k, d2, covarj, covark, dmax):
    """
    Second derivative w.r.t. two covariates.

    Parameters:
    d, n, score, dmat, dmax : as above
    d1j, d1k : arrays - First deriv memo for j and k.
    d2 : array - Second deriv memo.
    covarj, covark : arrays - Covariates j and k.

    Returns:
    float
    """
    idx = (n - 1) * dmax + d - 1
    if d2[idx] != NOTDONE:
        return d2[idx]
    val = coxd0(d - 1, n - 1, score, dmat, dmax) * score[n - 1] * covarj[n - 1] * covark[n - 1]
    if d < n:
        val += coxd2(d, n - 1, score, dmat, d1j, d1k, d2, covarj, covark, dmax)
    if d > 1:
        val += score[n - 1] * (
            coxd2(d - 1, n - 1, score, dmat, d1j, d1k, d2, covarj, covark, dmax) +
            covarj[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1k, covark, dmax) +
            covark[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1j, covarj, dmax)
        )
    d2[idx] = val
    return val

def exact_score_hessian(beta, X, y, strata, offset=None):
    """
    Compute exact score and Hessian for conditional logistic regression.

    Assumes one event per stratum (standard SCCS); for ties, extend with time grouping.

    Parameters:
    beta : array - Coefficients.
    X : array - Design matrix.
    y : array - Binary outcome (1 for event interval).
    strata : array - Stratum identifiers.
    offset : array, optional - Offset term (e.g., log(interval)).

    Returns:
    u : array - Score vector.
    imat : array - Hessian matrix (information matrix).
    """
    nvar = X.shape[1]
    u = np.zeros(nvar)
    imat = np.zeros((nvar, nvar))
    off = offset if offset is not None else np.zeros(len(X))

    for s in np.unique(strata):
        mask = strata == s
        X_s = X[mask]
        y_s = y[mask]
        off_s = off[mask]
        event_mask = y_s == 1
        if not event_mask.any():
            continue
        linpred = X_s @ beta + off_s
        score = np.exp(linpred)
        d0 = score.sum()
        prob = score / d0
        for j in range(nvar):
            u[j] += X_s[event_mask, j].sum() - (X_s[:, j] * prob).sum()
            for k in range(j + 1):
                imat[j, k] += -(X_s[:, j] * X_s[:, k] * prob).sum() + (X_s[:, j] * prob).sum() * (X_s[:, k] * prob).sum()
                imat[k, j] = imat[j, k]
    return u, imat

# Example usage
if __name__ == "__main__":
    # Simulated data for demonstration
    np.random.seed(42)
    n_indivs = 100
    intervals_per_indiv = 5
    n_total = n_indivs * intervals_per_indiv
    nvar = 3

    # Design matrix (random for demo)
    X = np.random.randn(n_total, nvar)
    # Outcome: one event per individual
    y = np.zeros(n_total)
    strata = np.repeat(np.arange(n_indivs), intervals_per_indiv)
    for i in range(n_indivs):
        event_idx = np.random.choice(np.where(strata == i)[0])
        y[event_idx] = 1
    # Offset: log(interval length), assume length 1 for simplicity
    offset = np.zeros(n_total)

    # Initial beta
    beta = np.zeros(nvar)

    # Compute exact score and Hessian
    u, imat = exact_score_hessian(beta, X, y, strata, offset)
    print("Score vector:", u)
    print("Hessian matrix:\n", imat)

    # For Newton-Raphson: update beta += np.linalg.solve(imat, u)
    # Repeat until convergence.

    print("\nImplement Newton-Raphson loop yourself for full fitting!")