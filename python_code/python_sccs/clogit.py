"""
Conditional Logistic Regression Implementation

This module provides a custom implementation of conditional logistic regression
using maximum likelihood estimation, matching the approach of R's survival::clogit.

For exact Hessian computation (as in R's C code for tied events), see the ported
functions below. In standard SCCS (one event per stratum), the approximation suffices.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

NOTDONE = -1.1  # Sentinel for uncomputed values in recursive functions

def conditional_loglik(beta, X, y, strata, offset=None):
    """
    Conditional log-likelihood for strata (individuals).
    
    Parameters:
    beta : array-like
        Coefficients.
    X : array-like
        Design matrix.
    y : array-like
        Binary outcome (1 for event interval).
    strata : array-like
        Stratum identifiers.
    offset : array-like, optional
        Offset term (e.g., log(interval)).
    
    Returns:
    float
        Negative log-likelihood.
    """
    ll = 0
    for s in np.unique(strata):
        mask = strata == s
        X_s = X[mask]
        y_s = y[mask]
        off_s = offset[mask] if offset is not None else np.zeros(len(X_s))
        if y_s.sum() == 1:  # One event per stratum
            logits = X_s @ beta + off_s
            ll += logits[y_s == 1] - logsumexp(logits)
    return -ll  # Negative for minimization

# Ported from R's survival C code for exact conditional logistic Hessian
# These functions compute exact derivatives for strata with tied events.
# In SCCS, usually no ties, so approximation is sufficient.

def coxd0(d, n, score, dmat, dmax):
    """
    Compute the exact denominator for conditional likelihood (recursive).

    Parameters:
    d : int
        Number of deaths (tied at time).
    n : int
        Number at risk.
    score : array
        Exponential terms (exp(zbeta)).
    dmat : array
        Memoization array (flattened, size n*dmax).
    dmax : int
        Max deaths in stratum.

    Returns:
    float
        Partial sum.
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
    d1 : array
        Memoization for first deriv (size n*dmax).
    covar : array
        Covariate values for stratum.

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
    d1j, d1k : arrays
        First deriv memo for j and k.
    d2 : array
        Second deriv memo.
    covarj, covark : arrays
        Covariates j and k.

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

def concordance_index(X, y, strata, beta, offset=None):
    """
    Compute concordance index for conditional logistic regression.
    
    Parameters:
    X : array-like
        Design matrix.
    y : array-like
        Binary outcome.
    strata : array-like
        Stratum identifiers.
    beta : array-like
        Fitted coefficients.
    offset : array-like, optional
        Offset term.
    
    Returns:
    float
        Concordance index.
    """
    off = offset if offset is not None else np.zeros(len(X))
    logits = X @ beta + off
    concordant = 0
    total_pairs = 0
    for s in np.unique(strata):
        mask = strata == s
        logits_s = logits[mask]
        y_s = y[mask]
        if y_s.sum() == 1:
            event_idx = np.where(y_s == 1)[0][0]
            event_logit = logits_s[event_idx]
            control_logits = logits_s[y_s == 0]
            for cl in control_logits:
                total_pairs += 1
                if event_logit > cl:
                    concordant += 1
    return concordant / total_pairs if total_pairs > 0 else np.nan

def exact_score_hessian(beta, X, y, strata, offset=None):
    """
    Compute exact score and Hessian for conditional logistic regression.

    Assumes one event per stratum (standard SCCS); for ties, would need extension.
    """
    nvar = X.shape[1]
    u = np.zeros(nvar)
    imat = np.zeros((nvar, nvar))
    for s in np.unique(strata):
        mask = strata == s
        X_s = X[mask]
        y_s = y[mask]
        off_s = offset[mask] if offset is not None else np.zeros(len(X_s))
        event_mask = y_s == 1
        if not event_mask.any():
            continue
        linpred = X_s @ beta + off_s
        # Stabilize probabilities using a max-shifted softmax
        max_lp = np.max(linpred)
        score = np.exp(linpred - max_lp)
        d0 = score.sum()
        prob = score / d0
        for j in range(nvar):
            u[j] += X_s[event_mask, j].sum() - (X_s[:, j] * prob).sum()
            for k in range(j + 1):
                imat[j, k] += -(X_s[:, j] * X_s[:, k] * prob).sum() + (X_s[:, j] * prob).sum() * (X_s[:, k] * prob).sum()
                imat[k, j] = imat[j, k]
    return u, imat

def fit_clogit(X, y, strata, offset=None, initial_beta=None, exact_hessian=False):
    """
    Fit conditional logistic regression.

    Parameters:
    X : array-like
        Design matrix.
    y : array-like
        Binary outcome.
    strata : array-like
        Stratum identifiers.
    offset : array-like, optional
        Offset term (e.g., log(interval)).
    initial_beta : array-like, optional
        Initial coefficients.
    exact_hessian : bool, optional
        Use exact Hessian computation (for tied events; default False).

    Returns:
    dict
        Fitted model results.
    """
    if initial_beta is None:
        initial_beta = np.zeros(X.shape[1])
    
    if exact_hessian:
        # Newton-Raphson with exact Hessian + damping and fallback
        beta = initial_beta.copy()
        maxiter = 100
        eps = 1e-8
        success = False
        message = "Exact Newton-Raphson did not converge"
        hess_inv = None
        ll_model_fun = conditional_loglik(beta, X, y, strata, offset)
        for _ in range(maxiter):
            u, imat = exact_score_hessian(beta, X, y, strata, offset)
            if not np.isfinite(u).all() or not np.isfinite(imat).all():
                break
            try:
                cond = np.linalg.cond(imat)
                delta = np.linalg.solve(imat, u)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(delta).all() or cond > 1e12 or np.max(np.abs(delta)) > 50:
                break
            step = 1.0
            new_ll = ll_model_fun
            while step > 1e-4:
                beta_new = beta + step * delta
                new_ll = conditional_loglik(beta_new, X, y, strata, offset)
                if np.isfinite(new_ll) and new_ll < ll_model_fun:
                    break
                step *= 0.5
            if step <= 1e-4:
                break
            beta = beta_new
            ll_model_fun = new_ll
            if np.max(np.abs(step * delta)) < eps:
                success = True
                message = "Exact Newton-Raphson converged"
                break
        if success:
            try:
                hess_inv = np.linalg.inv(imat)
            except:
                hess_inv = None
            params = beta
        else:
            args = (X, y, strata, offset)
            result = minimize(conditional_loglik, initial_beta, args=args, method='L-BFGS-B')
            params = result.x
            success = result.success
            message = f"Fallback to L-BFGS-B: {result.message}"
            hess_inv = getattr(result, 'hess_inv', None)
            if hess_inv is not None and hasattr(hess_inv, "todense"):
                hess_inv = np.asarray(hess_inv.todense())
            ll_model_fun = result.fun
    else:
        args = (X, y, strata, offset)
        result = minimize(conditional_loglik, initial_beta, args=args, method='L-BFGS-B')
        params = result.x
        success = result.success
        message = result.message
        hess_inv = getattr(result, 'hess_inv', None)
        if hess_inv is not None and hasattr(hess_inv, "todense"):
            hess_inv = np.asarray(hess_inv.todense())
        ll_model_fun = result.fun

    # Compute concordance
    conc = concordance_index(X, y, strata, params, offset)
    
    # Compute null log-likelihood
    ll_null = 0
    for s in np.unique(strata):
        mask = strata == s
        n_s = mask.sum()
        if n_s > 1:
            ll_null += -np.log(n_s)  # Since 1 event, prob 1/n_s
    
    ll_model = -ll_model_fun
    lr_stat = -2 * (ll_null - ll_model)
    df_lr = len(params)
    from scipy.stats import chi2
    p_lr = 1 - chi2.cdf(lr_stat, df_lr)

    # SEs and tests
    if hess_inv is not None:
        se = np.sqrt(np.diag(hess_inv))
        wald_stat = (params / se) ** 2
        p_wald = 1 - chi2.cdf(wald_stat, 1)
        z = params / se
        # Score test: approximation
        score_stat = wald_stat.sum()
        df_score = len(params)
        p_score = 1 - chi2.cdf(score_stat, df_score)
        # CI for exp(coef)
        ci_lower = np.exp(params - 1.96 * se)
        ci_upper = np.exp(params + 1.96 * se)
        wald_overall = wald_stat.sum()
        df_wald = len(params)
        p_wald_overall = 1 - chi2.cdf(wald_overall, df_wald)
    else:
        se = wald_stat = p_wald = z = ci_lower = ci_upper = np.full(len(params), np.nan)
        score_stat = p_score = wald_overall = p_wald_overall = np.nan
        df_score = df_wald = len(params)

    return {
        'params': params,
        'se': se,
        'z': z,
        'wald_stat': wald_stat,
        'p_wald': p_wald,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'success': success,
        'message': message,
        'loglik': ll_model,
        'concordance': conc,
        'll_null': ll_null,
        'lr_stat': lr_stat,
        'df_lr': df_lr,
        'p_lr': p_lr,
        'score_stat': score_stat,
        'df_score': df_score,
        'p_score': p_score,
        'wald_overall': wald_overall,
        'df_wald': df_wald,
        'p_wald_overall': p_wald_overall,
        'n_params': len(params),
        'hess_inv': hess_inv
    }