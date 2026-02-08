import pandas as pd
import numpy as np
from .adrug_matrix import adrug_matrix

def formatdata(indiv, astart, aend, aevent, adrug, aedrug, expogrp=None, washout=None, 
               sameexpopar=None, agegrp=None, seasongrp=None, dob=None, cov=None, 
               dataformat="stack", data=None):
    """
    Format data for SCCS analysis by creating chopped intervals.
    """
    # Process inputs
    adrug = [np.array(ad) for ad in adrug] if isinstance(adrug, list) else [np.array(adrug)]
    aedrug = [np.array(ae) for ae in aedrug] if isinstance(aedrug, list) else [np.array(aedrug)]
    
    for i in range(len(adrug)):
        adrug[i] -= 1
    for i in range(len(aedrug)):
        aedrug[i] = np.array(aedrug[i])
    
    if expogrp is None:
        expogrp = [0]
    washout = washout or [[] for _ in adrug]
    sameexpopar = sameexpopar or [True] * len(adrug)
    
    if expogrp is None or len(expogrp) == 0:
        expogrp = [[0] for _ in adrug]
    elif len(adrug) == 1 and len(expogrp) > 1:
        expogrp = [expogrp]
    else:
        for i in range(len(adrug)):
            expogrp[i] = expogrp[i]
    
    if not washout or len(washout) == 0:
        washout = [[0] for _ in adrug]
    elif len(adrug) == 1 and len(washout) > 1:
        washout = [np.array(washout) - 1]
    else:
        for i in range(len(adrug)):
            w = np.array(washout[i])
            if w.ndim == 0:
                w = np.array([w])
            washout[i] = w - 1 if w.size > 0 else np.array([])
    
    if agegrp is not None:
        agegrp = np.array(agegrp) - 1
    
    # data1
    data1 = pd.DataFrame({'indiv': indiv, 'aevent': aevent, 'astart': astart, 'aend': aend})
    if cov is not None:
        data1 = pd.concat([data1, cov], axis=1)
    if dob is not None:
        data1['dob'] = pd.to_datetime(dob.astype(str).str.zfill(8), format='%d%m%Y')
    data1 = data1.drop_duplicates().sort_values('indiv')
    data1['astart'] -= 1
    
    # adrug_all, aedrug_all
    if dataformat == "stack":
        adrug_all = [adrug_matrix(indiv, aevent, ad) for ad in adrug]
        aedrug_all = [adrug_matrix(indiv, aevent, ae) for ae in aedrug]
    elif dataformat == "multi":
        adrug_all = []
        for ad in adrug:
            df = pd.DataFrame({'indiv': indiv, 'aevent': aevent, 'adrug': ad.flatten()})
            df = df.sort_values(['indiv', 'adrug'])
            adrug_all.append(df['adrug'].values.reshape(len(data1), -1))
        aedrug_all = []
        for ae in aedrug:
            df = pd.DataFrame({'indiv': indiv, 'aevent': aevent, 'aedrug': ae.flatten()})
            df = df.sort_values(['indiv', 'aedrug'])
            aedrug_all.append(df['aedrug'].values.reshape(len(data1), -1))
    
    # expo1, expo2, expo
    expo1 = [np.full((ad.shape[0], ad.shape[1] * len(expogrp[i])), np.nan) for i, ad in enumerate(adrug_all)]
    for i in range(len(adrug)):
        for k in range(adrug_all[i].shape[1]):
            for j in range(len(expogrp[i])):
                expo1[i][:, j + len(expogrp[i]) * k] = adrug_all[i][:, k] + expogrp[i][j]
    
    expo2 = [np.full((ae.shape[0], ae.shape[1] * len(washout[i])), np.nan) for i, ae in enumerate(aedrug_all)]
    for i in range(len(adrug)):
        for k in range(aedrug_all[i].shape[1]):
            for j in range(len(washout[i])):
                expo2[i][:, j + len(washout[i]) * k] = aedrug_all[i][:, k] + washout[i][j]
    
    expo = []
    for i in range(len(adrug)):
        e = np.full((adrug_all[i].shape[0], adrug_all[i].shape[1] * (len(expogrp[i]) + len(washout[i]))), np.nan)
        for k in range(adrug_all[i].shape[1]):
            start = k * (len(expogrp[i]) + len(washout[i]))
            e[:, start:start + len(expogrp[i])] = expo1[i][:, k*len(expogrp[i]):(k+1)*len(expogrp[i])]
            e[:, start + len(expogrp[i]):start + len(expogrp[i]) + len(washout[i])] = expo2[i][:, k*len(washout[i]):(k+1)*len(washout[i])]
        expo.append(e)
    
    # Adjust expo
    for e in expo:
        for j in range(e.shape[1]):
            nan_mask = np.isnan(e[:, j])
            e[nan_mask, j] = data1['aend'].values[nan_mask]
        for k in range(1, e.shape[1]):
            e[:, k-1] = np.where(e[:, k] < e[:, k-1], e[:, k], e[:, k-1])
        e[:] = np.clip(e, data1['astart'].values[:, np.newaxis], data1['aend'].values[:, np.newaxis])
    
    # expolev
    expolev = []
    for i in range(len(adrug)):
        if dataformat == "stack":
            levels = list(range(1, len(expogrp[i]) + len(washout[i]))) + [0]
            expolev.append(np.tile(levels, adrug_all[i].shape[1]))
        else:
            levels = list(range(1, len(expogrp[i]) + len(washout[i]))) + [0]
            if sameexpopar[i]:
                expolev.append(np.tile(levels, adrug_all[i].shape[1]))
            else:
                # Simplified
                expolev.append(np.arange(len(levels) * adrug_all[i].shape[1]) % len(levels))
    
    # seasongrpY2
    seasongrpY2 = None
    if seasongrp is not None:
        years = np.arange(data1['dob'].min().year, (data1['dob'] + pd.to_timedelta(data1['aend'], unit='D')).max().year + 1)
        seasongrpY1 = []
        for y in years:
            for s in seasongrp:
                seasongrpY1.append(pd.to_datetime(f"{s:06d}{y:04d}", format='%m%d%Y'))
        seasongrpY1 = sorted(seasongrpY1)
        seasongrpY2 = np.full((len(data1), len(seasongrpY1)), np.nan)
        for j in range(len(seasongrpY1)):
            seasongrpY2[:, j] = (seasongrpY1[j] - data1['dob']).dt.days - 1
            seasongrpY2[:, j] = np.clip(seasongrpY2[:, j], data1['astart'], data1['aend'])
    
    ncolexpo = [e.shape[1] for e in expo]
    ncuts = sum(ncolexpo) + (len(agegrp) if agegrp is not None else 0) + 2 + (seasongrpY2.shape[1] if seasongrpY2 is not None else 0)
    nevents = len(data1)
    ind = np.repeat(np.arange(1, nevents + 1), ncuts)
    eventday = np.repeat(data1['aevent'].values, ncuts)
    allexpo = np.concatenate(expo, axis=1)
    
    cutp_parts = [data1['astart'], data1['aend'], allexpo.flatten()]
    if agegrp is not None:
        cutp_parts.append(np.tile(agegrp, nevents))
    if seasongrpY2 is not None:
        cutp_parts.append(seasongrpY2.flatten())
    cutp = np.concatenate(cutp_parts)
    
    sort_idx = np.lexsort((cutp, ind))
    ind, cutp, eventday = ind[sort_idx], cutp[sort_idx], eventday[sort_idx]
    
    interval = np.concatenate([[0], np.diff(cutp)])
    # Reset interval at individual boundaries to avoid negative diffs
    boundary_mask = np.concatenate([[True], ind[1:] != ind[:-1]])
    interval[boundary_mask] = 0
    # Guard against any remaining negative values
    interval[interval < 0] = 0
    interval[cutp <= data1['astart'].iloc[ind - 1].values] = 0
    interval[cutp > data1['aend'].iloc[ind - 1].values] = 0
    
    event = (data1['aevent'].iloc[ind - 1].values > cutp - interval) & (data1['aevent'].iloc[ind - 1].values <= cutp)
    
    season = None
    if seasongrp is not None:
        cutp_dates = data1['dob'].iloc[ind - 1] + pd.to_timedelta(cutp, unit='D')
        cutpseason = cutp_dates.dt.strftime('%m/%d')
        seasoncutpts = pd.Series(seasongrpY1).dt.strftime('%m/%d').unique()
        season = np.full(len(cutp), len(seasongrp))
        for i, pt in enumerate(seasoncutpts):
            season = np.where(cutpseason >= pt, i + 1, season)
        season = pd.Categorical(season)
    
    age_bins = sorted([data1['astart'].min()] + (agegrp.tolist() if agegrp is not None else []) + [data1['aend'].max()])
    agegr = pd.cut(cutp, bins=age_bins, labels=False, include_lowest=True, duplicates='drop') + 1  # Match R's 1-based indexing
    agegr = pd.Categorical(agegr)
    
    exgr = np.zeros((len(cutp), len(adrug)))
    for i in range(len(adrug)):
        for k in range(expo[i].shape[1]):
            exgr[:, i] = np.where(cutp > expo[i][ind - 1, k], expolev[i][k], exgr[:, i])
    exgr_df = pd.DataFrame(exgr, columns=[f"drug_{i}" for i in range(len(adrug))])
    for col in exgr_df.columns:
        exgr_df[col] = pd.Categorical(exgr_df[col])
    exgr1 = pd.concat([exgr_df, pd.DataFrame({'trial': 1}, index=exgr_df.index)], axis=1)
    
    # chopdat
    mask = interval != 0
    chopdat_dict = {
        'indivL': pd.Categorical(ind[mask]),
        'event': event[mask].astype(int),
        'eventday': eventday[mask],
        'lower': cutp[mask] - interval[mask] + 1,
        'upper': cutp[mask],
        'interval': interval[mask],
        'age': agegr[mask]
    }
    if season is not None:
        chopdat_dict['season'] = season[mask]
    chopdat = pd.DataFrame(chopdat_dict, index=None)  # Avoid index issues
    exgr1_selected = exgr1.loc[mask].copy()  # Use .loc for boolean indexing, copy to avoid issues
    exgr1_selected = exgr1_selected.reset_index(drop=True)
    chopdat = chopdat.reset_index(drop=True)
    chopdat = pd.concat([chopdat, exgr1_selected], axis=1, ignore_index=False).drop(columns=['trial'])
    chopdat = chopdat.reset_index(drop=True)  # Final reset to ensure clean index
    
    # Repeat data1 with proper indexing
    indiv_indices = ind[mask] - 1  # 0-based indices for data1
    for col in data1.columns:
        chopdat[col] = data1[col].iloc[indiv_indices].values
    
    chopdat['astart'] += 1
    return chopdat