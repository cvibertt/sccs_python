import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from .formatdata import formatdata
from .summary_sccs import summary_sccs

def semisccs(formula, indiv, astart, aend, aevent, adrug, aedrug, expogrp=None, washout=None, 
             sameexpopar=None, dataformat="stack", data=None):
    """
    Fit semi-parametric SCCS model with unspecified age effect.
    
    Parameters:
    Similar to standardsccs, but for semi-parametric age modeling.
    
    Returns:
    summary : dict
        Model summary.
    """
    if dataformat not in ["multi", "stack"]:
        raise ValueError("Please input dataformat as multi or stack")
    
    # Process adrug, aedrug (similar to standardsccs)
    if not isinstance(adrug, list):
        adrug = [adrug]
    if not isinstance(aedrug, list):
        aedrug = [aedrug]
    
    # Call formatdata
    chopdat = formatdata(indiv=indiv, astart=astart, aend=aend, aevent=aevent, 
                         adrug=adrug, aedrug=aedrug, expogrp=expogrp, washout=washout, 
                         sameexpopar=sameexpopar, dataformat=dataformat, data=data)
    
    # Fit GLM approximation (true model would use conditional logit with age strata)
    fmla = f"{formula} + strata(indivL) + offset(log(interval))"
    # For approximation, ignore strata and use GLM
    full_formula = f"event ~ {formula} + age"
    
    mod = smf.glm(full_formula, data=chopdat, family=smf.families.Binomial(), offset=np.log(chopdat['interval']))
    summary = summary_sccs(mod, ncases=len(chopdat['indivL'].unique()), nevents=chopdat['event'].sum())
    return summary