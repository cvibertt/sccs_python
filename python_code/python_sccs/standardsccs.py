import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .formatdata import formatdata

def standardsccs(formula, indiv, astart, aend, aevent, adrug, aedrug, expogrp=None, washout=None, 
                 sameexpopar=None, agegrp=None, seasongrp=None, dob=None, dataformat="stack", data=None):
    if data is not None and dataformat == 'stack':
        chopdat = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    else:
        if data is not None:
            # Extract columns directly, matching R's eval(substitute(...))
            data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            indiv = data_df[indiv].values if indiv is not None else indiv
            astart = data_df[astart].values if astart is not None else astart
            aend = data_df[aend].values if aend is not None else aend
            aevent = data_df[aevent].values if aevent is not None else aevent
            adrug = data_df[adrug].values if adrug is not None else adrug
            aedrug = data_df[aedrug].values if aedrug is not None else aedrug
            dataformat = "stack"  # Use stack format, as in R
        # If expogrp is None, set to unique adrug values
        if expogrp is None and adrug is not None:
            expogrp = np.sort(np.unique(adrug))
        # Ensure expogrp is list of lists
        if isinstance(expogrp, list) and len(expogrp) > 0 and isinstance(expogrp[0], (int, float)):
            expogrp = [expogrp]
        chopdat = formatdata(indiv=indiv, astart=astart, aend=aend, aevent=aevent, adrug=adrug, aedrug=aedrug, 
                             expogrp=expogrp, washout=washout, sameexpopar=sameexpopar, agegrp=agegrp, 
                             seasongrp=seasongrp, dob=dob, dataformat=dataformat, data=data)

    
    # Fit conditional logistic regression
    from patsy import dmatrix
    from .clogit import fit_clogit
    
    # Create design matrix (drop global intercept for identifiability)
    X = dmatrix(formula, chopdat, return_type='dataframe')
    if 'Intercept' in X.columns:
        X = X.drop(columns=['Intercept'])
    coef_names = X.columns.tolist()
    X = X.values
    y = chopdat['event'].values.astype(int)
    strata = pd.Categorical(chopdat['indivL']).codes
    offset = np.log(np.maximum(chopdat['interval'].values, 1e-10))  # Add offset like R, avoid log(0)
    
    # Fit
    result = fit_clogit(X, y, strata, offset=offset, exact_hessian=True)

    # Compute overall Wald
    from scipy.stats import chi2
    overall_wald = np.sum(result['wald_stat'])
    df_wald = len(result['params'])
    p_overall_wald = 1 - chi2.cdf(overall_wald, df_wald)

    # Create result object
    class ClogitResult:
        def __init__(self, params, success, message, loglik, concordance, lr_stat, df_lr, p_lr, se, z, wald_stats, p_wald, overall_wald, df_wald, p_overall_wald, score_stat, df_score, p_score, ci_lower, ci_upper, n_params, n_events, n_total, coef_names):
            self.params = params
            self.success = success
            self.message = message
            self.llf = loglik
            self.concordance = concordance
            self.lr_stat = lr_stat
            self.df_lr = df_lr
            self.p_lr = p_lr
            self.se = se
            self.z = z
            self.wald_stats = wald_stats
            self.p_wald = p_wald
            self.wald_overall = overall_wald
            self.df_wald = df_wald
            self.p_overall_wald = p_overall_wald
            self.score_stat = score_stat
            self.df_score = df_score
            self.p_score = p_score
            self.ci_lower = ci_lower
            self.ci_upper = ci_upper
            self.df_model = n_params
            self.pvalues = p_wald
            self.conf_int = np.column_stack((ci_lower, ci_upper))
            self.n_events = n_events
            self.n_total = n_total
            self.coef_names = coef_names

        def __str__(self):
            # Mimic R's coxph output
            lines = []
            lines.append("Call:")
            lines.append("coxph(formula = Surv(rep(1, {}L), event) ~ ... + strata(indivL) +".format(self.n_total))
            lines.append("      offset(log(interval)), data = chopdat, method = \"exact\")")
            lines.append("")
            lines.append("n= {}, number of events= {}".format(self.n_total, self.n_events))
            lines.append("")
            # Coefficient table
            lines.append("coef exp(coef) se(coef) z Pr(>|z|)")
            for i in range(len(self.params)):
                coef = self.params[i]
                exp_coef = np.exp(coef)
                se = self.se[i]
                z = self.z[i]
                p = self.p_wald[i]
                star = ""
                if p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                elif p < 0.1:
                    star = "."
                lines.append(f"{float(coef):.4f} {float(exp_coef):.4f} {float(se):.4f} {float(z):.3f} {float(p):.2e}{star}")
            lines.append("---")
            lines.append("Signif. codes: 0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
            lines.append("")
            # CI table
            lines.append("exp(coef) exp(-coef) lower .95 upper .95")
            for i in range(len(self.params)):
                exp_coef = np.exp(self.params[i])
                exp_neg = 1 / exp_coef
                lower = self.ci_lower[i]
                upper = self.ci_upper[i]
                lines.append(f"{float(exp_coef):.4f} {float(exp_neg):.4f} {float(lower):.3f} {float(upper):.3f}")
            lines.append("")
            lines.append(f"Concordance= {float(self.concordance):.3f} (se = NA)")
            lines.append(f"Likelihood ratio test= {float(self.lr_stat):.1f} on {int(self.df_lr)} df, p={float(self.p_lr):.2e}")
            lines.append(f"Wald test = {float(self.wald_overall):.1f} on {int(self.df_wald)} df, p={float(self.p_overall_wald):.2e}")
            lines.append(f"Score (logrank) test = {float(self.score_stat):.1f} on {int(self.df_score)} df, p={float(self.p_score):.2e}")
            return "\n".join(lines)

    n_events = chopdat['event'].sum()
    n_total = len(chopdat)

    return ClogitResult(result['params'], result['success'], result['message'], result['loglik'], result['concordance'], result['lr_stat'], result['df_lr'], result['p_lr'], result['se'], result['z'], result['wald_stat'], result['p_wald'], result['wald_overall'], result['df_wald'], result['p_wald_overall'], result['score_stat'], result['df_score'], result['p_score'], result['ci_lower'], result['ci_upper'], result['n_params'], n_events, n_total, coef_names)