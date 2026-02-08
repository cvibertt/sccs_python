import numpy as np
import pandas as pd


def build_time_grid(start, end, bin_size=1.0):
    """
    Build left-closed, right-open bins [t_k, t_{k+1}) for an interval.
    Returns bin edges and number of bins.
    """
    if end <= start:
        raise ValueError("end must be greater than start.")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    n_bins = int(np.ceil((end - start) / bin_size))
    edges = start + bin_size * np.arange(n_bins + 1)
    return edges, n_bins


def bin_events(event_times, start, end, bin_size=1.0):
    """
    Bin event timestamps into counts per interval.
    """
    edges, n_bins = build_time_grid(start, end, bin_size)
    if event_times is None or len(event_times) == 0:
        return np.zeros(n_bins, dtype=int)
    event_times = np.asarray(event_times, dtype=float)
    mask = (event_times >= start) & (event_times < end)
    counts, _ = np.histogram(event_times[mask], bins=edges)
    return counts.astype(int)


def bin_point_exposures(exposure_times, start, end, bin_size=1.0):
    """
    Bin point exposures into a binary vector (1 if an exposure starts in the bin).
    """
    edges, n_bins = build_time_grid(start, end, bin_size)
    if exposure_times is None or len(exposure_times) == 0:
        return np.zeros(n_bins, dtype=int)
    exposure_times = np.asarray(exposure_times, dtype=float)
    mask = (exposure_times >= start) & (exposure_times < end)
    counts, _ = np.histogram(exposure_times[mask], bins=edges)
    return (counts > 0).astype(int)


def build_age_design(edges, age_bins):
    """
    Build a one-hot age-group design matrix based on bin midpoints.
    age_bins should be a sorted list of cutpoints in the same unit as edges.
    """
    if age_bins is None or len(age_bins) == 0:
        raise ValueError("age_bins must be provided.")
    edges = np.asarray(edges, dtype=float)
    age_bins = np.asarray(age_bins, dtype=float)
    mids = (edges[:-1] + edges[1:]) / 2.0
    bins = np.concatenate(([mids.min() - 1e-6], age_bins, [mids.max() + 1e-6]))
    cats = pd.cut(mids, bins=bins, labels=False, include_lowest=True)
    g = int(np.nanmax(cats)) + 1
    design = np.zeros((len(mids), g), dtype=float)
    for i, c in enumerate(cats):
        if pd.isna(c):
            continue
        design[i, int(c)] = 1.0
    return design


def make_patient_arrays(
    events_df,
    exposures_df,
    id_col,
    start_col,
    end_col,
    event_time_col,
    exposure_time_col,
    drug_col,
    bin_size=1.0,
    global_start=None,
    global_end=None,
    age_bins=None,
):
    """
    Convert patient-level events and exposures to ConvSCCS input lists.

    events_df: one row per event with columns (id, event_time)
    exposures_df: one row per exposure with columns (id, drug, exposure_time)
    start_col/end_col: observation window bounds per patient (in events_df or exposures_df)

    Returns:
        y_list: list of (K,) arrays with event counts
        x_list: list of (d, K) arrays with point exposures
        drug_index: dict mapping drug ids to column indices
        patient_ids: list of patient ids in output order
    """
    if id_col not in events_df.columns:
        raise ValueError(f"{id_col} missing in events_df")
    if id_col not in exposures_df.columns:
        raise ValueError(f"{id_col} missing in exposures_df")

    # Map drugs to indices
    drugs = sorted(exposures_df[drug_col].dropna().unique().tolist())
    drug_index = {d: i for i, d in enumerate(drugs)}

    # Determine observation windows per patient
    obs_df = (
        events_df[[id_col, start_col, end_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)
    )

    # Global grid if provided (ensures same K for all patients)
    if global_start is None:
        global_start = float(obs_df[start_col].min())
    if global_end is None:
        global_end = float(obs_df[end_col].max())
    edges, n_bins = build_time_grid(global_start, global_end, bin_size)

    age_design_global = None
    if age_bins is not None:
        age_design_global = build_age_design(edges, age_bins)

    y_list = []
    x_list = []
    age_list = []
    patient_ids = []

    for pid, row in obs_df.iterrows():
        start = float(row[start_col])
        end = float(row[end_col])
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if end <= start:
            continue

        ev_times = events_df.loc[events_df[id_col] == pid, event_time_col].values
        # Bin on the global grid and zero out bins outside observation window
        y = np.zeros(n_bins, dtype=int)
        if ev_times is not None and len(ev_times) > 0:
            ev_times = np.asarray(ev_times, dtype=float)
            mask = (ev_times >= global_start) & (ev_times < global_end)
            y += np.histogram(ev_times[mask], bins=edges)[0].astype(int)

        X = np.zeros((len(drugs), n_bins), dtype=int)
        exp_rows = exposures_df.loc[exposures_df[id_col] == pid]
        for drug, idx in drug_index.items():
            drug_times = exp_rows.loc[exp_rows[drug_col] == drug, exposure_time_col].values
            if drug_times is not None and len(drug_times) > 0:
                drug_times = np.asarray(drug_times, dtype=float)
                mask = (drug_times >= global_start) & (drug_times < global_end)
                X[idx] = (np.histogram(drug_times[mask], bins=edges)[0] > 0).astype(int)

        # Zero out bins outside the patient observation window
        start_bin = int(np.floor((start - global_start) / bin_size))
        end_bin = int(np.ceil((end - global_start) / bin_size))
        start_bin = max(0, start_bin)
        end_bin = min(n_bins, end_bin)
        if start_bin > 0:
            y[:start_bin] = 0
            X[:, :start_bin] = 0
        if end_bin < n_bins:
            y[end_bin:] = 0
            X[:, end_bin:] = 0

        if age_design_global is not None:
            age_design = age_design_global.copy()
            if start_bin > 0:
                age_design[:start_bin, :] = 0
            if end_bin < n_bins:
                age_design[end_bin:, :] = 0
            age_list.append(age_design)

        y_list.append(y)
        x_list.append(X)
        patient_ids.append(pid)

    return y_list, x_list, age_list if age_list else None, drug_index, patient_ids
