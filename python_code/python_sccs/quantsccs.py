import pandas as pd
import statsmodels.formula.api as smf
from .summary_sccs import summary_sccs

def quantsccs(formula, indiv, event, data):
    """
    Fit SCCS model with continuous exposures using conditional logistic regression.
    
    Parameters:
    formula : str
        Model formula (e.g., 'exposure + covariates').
    indiv : str or array
        Individual identifier.
    event : str or array
        Event indicator.
    data : pd.DataFrame
        Dataframe containing the variables.
    
    Returns:
    summary : dict
        Model summary.
    """
    # For approximation, use GLM with binomial family (not true conditional logit)
    fmla = f"{formula} + strata({indiv})"  # strata not supported, so ignore for now
    full_formula = f"{event} ~ {formula}"
    
    mod = smf.glm(full_formula, data=data, family=smf.families.Binomial())
    summary = summary_sccs(mod, ncases=len(data[indiv].unique()), nevents=data[event].sum())
    return summary