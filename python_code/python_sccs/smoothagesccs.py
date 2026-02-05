import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from .formatdata import formatdata
from .dmsplinedesign import dmsplinedesign

def smoothagesccs(indiv, astart, aend, aevent, adrug, aedrug, expogrp=0, washout=None, 
                  kn=12, sp=None, data=None):
    """
    Fit smoothed age SCCS model.
    
    Parameters:
    Similar to nonparasccs, but focuses on age smoothing.
    
    Returns:
    result : dict
        Fitted model results.
    """
    # Data processing
    fdata = pd.DataFrame({
        'indiv': indiv,
        'startob': astart,
        'endob': aend,
        'st_risk': adrug,
        'end_risk': aedrug,
        'eventday': aevent
    })
    
    # Process st_risk, end_risk similar to nonparasccs
    fdata['st_risk'] = np.maximum(fdata['startob'], fdata['st_risk'])
    fdata['st_risk'] = np.minimum(fdata['endob'], fdata['st_risk'])
    fdata['end_risk'] = np.maximum(fdata['startob'], fdata['end_risk'])
    fdata['end_risk'] = np.minimum(fdata['endob'], fdata['end_risk'])
    
    knots1 = np.linspace(fdata['startob'].min(), fdata['endob'].max() + 0.0005, kn)
    
    # Optimization
    def neg_ll(params):
        beta_age = params[:-1] if len(params) > kn else params
        beta_expo = params[-1] if expogrp != 0 else 0
        ll = 0
        for idx, row in fdata.iterrows():
            startob = row['startob']
            endob = row['endob']
            st_risk = row['st_risk']
            end_risk = row['end_risk']
            eventday = row['eventday']
            
            # Compute integral
            integral = 0
            # Pre exposure
            if startob < st_risk:
                integral += quad(lambda t: np.exp(dmsplinedesign([t], knots1, 4)[0] @ beta_age), startob, st_risk)[0]
            # Exposure
            if st_risk < end_risk:
                integral += quad(lambda t: np.exp(dmsplinedesign([t], knots1, 4)[0] @ beta_age + beta_expo), st_risk, end_risk)[0]
            # Post
            if end_risk < endob:
                integral += quad(lambda t: np.exp(dmsplinedesign([t], knots1, 4)[0] @ beta_age), end_risk, endob)[0]
            
            if pd.isna(eventday):
                ll -= integral
            else:
                # log rate at event
                log_rate_event = dmsplinedesign([eventday], knots1, 4)[0] @ beta_age
                if st_risk <= eventday <= end_risk:
                    log_rate_event += beta_expo
                ll += log_rate_event - integral
        
        return -ll
    
    n_params = kn + (1 if expogrp != 0 else 0)
    initial_params = np.zeros(n_params)
    result = minimize(neg_ll, initial_params, method='L-BFGS-B')
    
    # Return structured result
    return {
        'coefficients': result.x,
        'knots': knots1,
        'smoothingpara': sp
    }