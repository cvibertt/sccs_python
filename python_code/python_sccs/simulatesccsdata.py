import numpy as np
import pandas as pd

def simulatesccsdata(nindivs, astart, aend, adrug, aedrug, eexpo, expogrp=None, washout=None, ewashout=None, agegrp=None, eage=None):
    """
    Simulate SCCS data.
    
    Parameters:
    nindivs : int
        Number of individuals.
    astart, aend : float or array
        Start and end ages.
    adrug, aedrug : array
        Drug start and end times.
    expogrp : list
        Exposure groups.
    eexpo : list or float
        Exposure relative incidences.
    washout, ewashout : optional
        Washout parameters.
    agegrp, eage : optional
        Age groups and effects.
    
    Returns:
    data : pd.DataFrame
        Simulated data.
    """
    if expogrp is None:
        expogrp = [0]
    
    if isinstance(eexpo, list) and len(expogrp) != len(eexpo):
        raise ValueError("Please provide true relative incidence value for each exposure group")
    
    if np.isscalar(astart):
        start = np.repeat(astart - 1, nindivs)
    else:
        start = np.array(astart) - 1
    
    if np.isscalar(aend):
        end = np.repeat(aend, nindivs)
    else:
        end = np.array(aend)
    
    adrug = np.array(adrug)
    aedrug = np.array(aedrug)
    
    if adrug.ndim == 1:
        adrug = np.tile(adrug, (nindivs, 1))
    adrug -= 1
    
    if aedrug.ndim == 1:
        aedrug = np.tile(aedrug, (nindivs, 1))
    
    for i in range(adrug.shape[1]):
        adrug[:, i] = np.maximum(adrug[:, i], start)
        adrug[:, i] = np.minimum(adrug[:, i], end)
    
    for i in range(aedrug.shape[1]):
        aedrug[:, i] = np.maximum(aedrug[:, i], start)
        aedrug[:, i] = np.minimum(aedrug[:, i], end)
    
    # Assume single drug for simplicity
    expo_start = adrug[:, 0]
    expo_end = aedrug[:, 0]
    
    # Baseline rate
    baseline_rate = 1.0
    expo_rate = eexpo[0] if isinstance(eexpo, list) else eexpo
    
    aevents = []
    for i in range(nindivs):
        obs_start = start[i]
        obs_end = end[i]
        e_start = expo_start[i]
        e_end = expo_end[i]
        
        # Define periods: pre, expo, post
        periods = [
            (obs_start, min(e_start, obs_end), baseline_rate),
            (max(e_start, obs_start), min(e_end, obs_end), expo_rate),
            (max(e_end, obs_start), obs_end, baseline_rate)
        ]
        
        # Calculate total lambda
        total_lambda = 0
        for p_start, p_end, rate in periods:
            if p_end > p_start:
                total_lambda += (p_end - p_start) * rate
        
        # Sample number of events
        num_events = np.random.poisson(total_lambda)
        
        if num_events > 0:
            # Sample event time proportionally
            u = np.random.uniform(0, total_lambda)
            cum_lambda = 0
            for p_start, p_end, rate in periods:
                if p_end > p_start:
                    cum_lambda += (p_end - p_start) * rate
                    if u <= cum_lambda:
                        aevent = np.random.uniform(p_start, p_end)
                        break
            else:
                aevent = obs_end  # fallback
        else:
            aevent = np.nan
        
        aevents.append(aevent)
    
    data = pd.DataFrame({
        'indiv': np.arange(1, nindivs + 1),
        'astart': start + 1,
        'aend': end,
        'aevent': aevents,
        'adrug': expo_start + 1,
        'aedrug': expo_end
    })
    
    return data