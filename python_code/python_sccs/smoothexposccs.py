import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from .dmsplinedesign import dmsplinedesign

def smoothexposccs(indiv, astart, aend, aevent, adrug, aedrug, agegrp, kn=12, sp=None, data=None):
    """
    Fit smoothed exposure SCCS model.
    
    Parameters:
    Similar to smoothagesccs, but for exposure smoothing.
    
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
    
    fdata['st_risk'] = np.maximum(fdata['startob'], fdata['st_risk'])
    fdata['st_risk'] = np.minimum(fdata['endob'], fdata['st_risk'])
    fdata['end_risk'] = np.maximum(fdata['startob'], fdata['end_risk'])
    fdata['end_risk'] = np.minimum(fdata['endob'], fdata['end_risk'])
    
    # Knots for age
    knots1 = np.linspace(fdata['startob'].min(), fdata['endob'].max() + 0.0005, kn)
    
    # Knots for exposure
    knots1ex = np.linspace(0, (fdata['end_risk'] - fdata['st_risk']).max() + 0.00001, kn)
    
    # Optimization
    def neg_ll(params):
        beta_age = params[:kn]
        beta_exp = params[kn:2*kn]
        ll = 0
        for idx, row in fdata.iterrows():
            startob = row['startob']
            endob = row['endob']
            st_risk = row['st_risk']
            end_risk = row['end_risk']
            eventday = row['eventday']
            
            # Define rate function
            def rate(t):
                age_design = dmsplinedesign(np.array([t]), knots1, 4)[0]
                log_rate = age_design @ beta_age
                if st_risk <= t <= end_risk:
                    timesince = t - st_risk
                    exp_design = dmsplinedesign(np.array([timesince]), knots1ex, 4)[0]
                    log_rate += exp_design @ beta_exp
                return np.exp(log_rate)
            
            # Compute integral
            integral, _ = quad(rate, startob, endob)
            
            if pd.isna(eventday):
                ll -= integral
            else:
                # log rate at event
                age_design = dmsplinedesign(np.array([eventday]), knots1, 4)[0]
                log_rate_event = age_design @ beta_age
                if st_risk <= eventday <= end_risk:
                    timesince = eventday - st_risk
                    exp_design = dmsplinedesign(np.array([timesince]), knots1ex, 4)[0]
                    log_rate_event += exp_design @ beta_exp
                ll += log_rate_event - integral
        
        return -ll
    
    initial_params = np.zeros(2 * kn)
    result = minimize(neg_ll, initial_params, method='L-BFGS-B')
    
    return {
        'coefficients': result.x,
        'knots_age': knots1,
        'knots_exp': knots1ex,
        'smoothingpara': sp
    }