import pandas as pd
import numpy as np

def max_expo(indiv, adrug):
    """
    Determine the maximum number of exposures per case
    """
    expo_no = pd.DataFrame({'indiv': indiv, 'adrug': adrug})
    expo_no_unique = expo_no.drop_duplicates()
    no_of_expo = expo_no_unique.groupby('indiv').size()
    max_no_of_expo = no_of_expo.max()
    return max_no_of_expo

def adrug_matrix(indiv, aevent, adrug):
    """
    A function that converts the adrug or aedrug column vector to a matrix with
    nrow = number of total events (not cases as some cases can have more than one event)
    and ncol=the maxmimum number of exposures per case.
    """
    events_no = pd.DataFrame({'indiv': indiv, 'aevent': aevent})
    events_no_unique = events_no.drop_duplicates()
    indiv_list = events_no_unique['indiv'].tolist()
    
    events_expo = pd.DataFrame({'indiv': indiv, 'aevent': aevent, 'adrug': adrug})
    
    # Sort by indiv and adrug
    events_expo = events_expo.sort_values(by=['indiv', 'adrug'])
    
    events_expo_unique = events_expo.drop_duplicates()
    
    # number of exposures for each case
    no_of_expo = events_expo_unique.groupby('indiv').size().reindex(indiv_list).fillna(0).astype(int)
    no_of_events = events_no_unique.groupby('indiv').size().reindex(indiv_list).fillna(0).astype(int)
    
    # The actual number of exposures for each case
    no_of_expo_case = no_of_expo / no_of_events.replace(0, 1)  # avoid div by 0
    
    # the actual number of exposures for each event
    no_of_expo_events = no_of_expo_case.values
    
    # Cumulative sum of the number of exposures per event
    cumsum_no_of_expo_events = np.cumsum(no_of_expo_events)
    cumsum_no_of_expo_events_1 = np.concatenate([[1], cumsum_no_of_expo_events[:-1] + 1])
    
    # max number of exposures, use the function
    max_no_of_expo = max_expo(indiv, adrug)
    adrug_new = np.full((len(indiv_list), max_no_of_expo), np.nan)
    
    for i in range(len(indiv_list)):
        start_idx = int(cumsum_no_of_expo_events_1[i]) - 1  # 0-based
        end_idx = int(cumsum_no_of_expo_events[i]) - 1
        num = int(no_of_expo_events[i])
        if num > 0 and start_idx <= end_idx:
            adrug_new[i, :num] = events_expo_unique.iloc[start_idx:end_idx+1, 2].values[:num]
    
    return adrug_new