import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from .dmsplinedesign import dmsplinedesign
from .msplinedesign import msplinedesign

def nonparasccs(indiv, astart, aend, aevent, adrug, aedrug, kn1=12, kn2=12, sp1=None, sp2=None, data=None):
    """
    Non-parametric SCCS method with spline functions for age and exposure effects.
    """
    # Data preparation
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
    fdata['st_risk'] = fdata['st_risk'].fillna(fdata['endob'])
    fdata['end_risk'] = fdata['end_risk'].fillna(fdata['endob'])
    fdata['end_risk'] = np.minimum(fdata['endob'], fdata['end_risk'])
    
    fdata['expostatus'] = ((fdata['eventday'] >= fdata['st_risk']) & 
                           (fdata['eventday'] <= fdata['end_risk']) & 
                           (fdata['st_risk'] != fdata['endob'])).astype(int)
    fdata['timesinceex'] = fdata['eventday'] - fdata['st_risk']
    fdata['timesinceex'] = np.maximum(0, fdata['timesinceex'])
    fdata['timesinceex'] = np.minimum(fdata['timesinceex'], fdata['end_risk'] - fdata['st_risk'])
    
    knots1 = np.linspace(fdata['startob'].min(), fdata['endob'].max() + 0.0005, kn1)
    
    data4 = fdata.copy()
    data4['timesincevac'] = data4['eventday'] - data4['st_risk']
    data4['risklen'] = data4['end_risk'] - data4['st_risk']
    data4['expostatus'] = ((data4['timesincevac'] >= 0) & (data4['timesincevac'] <= data4['risklen'])).astype(int)
    timesincevacwithinexpo = data4.loc[data4['expostatus'] == 1, 'timesincevac'].values
    timesincevacwithinexpo = np.concatenate([timesincevacwithinexpo, [0, data4['risklen'].max() + 0.00001]])
    knots1ex = np.linspace(timesincevacwithinexpo.min(), timesincevacwithinexpo.max(), kn2)

    # Determine basis sizes from spline design matrices
    n_basis_age = dmsplinedesign(np.array([fdata['startob'].min()]), knots1, 4).shape[1]
    n_basis_exp = dmsplinedesign(np.array([0.0]), knots1ex, 4).shape[1]
    
    # Optimization
    def neg_ll(params):
        beta_age = params[:n_basis_age]
        beta_exp = params[n_basis_age:n_basis_age + n_basis_exp]
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
        
        return -ll  # negative log likelihood
    
    initial_params = np.zeros(n_basis_age + n_basis_exp)
    result = minimize(neg_ll, initial_params, method='L-BFGS-B')
    
    # Return fitted parameters or summary
    return result