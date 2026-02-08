import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from convsccs import ConvSCCS
from data_utils import make_patient_arrays


def main():
    apdat = pd.read_csv("../python_code/apdat.csv")

    # Build events and exposures tables from apdat
    events_df = apdat[["case", "sta", "end", "stro"]].drop_duplicates()
    base_expo = apdat[["case", "ap", "endap"]].dropna().drop_duplicates()
    base_expo = base_expo.rename(columns={"ap": "exposure_time"})
    base_expo["drug"] = "ap"

    # Add washout exposures at endap + w
    washout = [1, 92, 182]
    washout_expo = []
    for w in washout:
        w_expo = base_expo.copy()
        w_expo["exposure_time"] = w_expo["endap"] + w
        w_expo["drug"] = f"ap_w{w}"
        washout_expo.append(w_expo[["case", "exposure_time", "drug"]])

    exposures_df = pd.concat(
        [base_expo[["case", "exposure_time", "drug"]]] + washout_expo,
        ignore_index=True,
    )

    global_start = float(apdat["sta"].min())
    global_end = float(apdat["end"].max())

    age_years = list(range(40, 91, 10))
    agedem = [int(np.floor(age * 365.25)) for age in age_years]

    y_list, x_list, age_list, drug_index, patient_ids = make_patient_arrays(
        events_df=events_df,
        exposures_df=exposures_df,
        id_col="case",
        start_col="sta",
        end_col="end",
        event_time_col="stro",
        exposure_time_col="exposure_time",
        drug_col="drug",
        bin_size=7.0,
        global_start=global_start,
        global_end=global_end,
        age_bins=agedem,
    )

    print("ConvSCCS apdat trial (weekly bins, 182-day risk window)", flush=True)
    print(f"Patients: {len(patient_ids)}", flush=True)
    print(f"Drugs: {drug_index}", flush=True)
    print(f"Example y shape: {y_list[0].shape}, X shape: {x_list[0].shape}", flush=True)
    print(f"Age bins (years): {age_years}", flush=True)

    model = ConvSCCS(
        p=26,
        gamma_tv=0.1,
        gamma_gl=0.1,
        max_iter=3,
        step_size=1.0,
        verbose=True,
        use_phi=False,
    )
    model.fit(y_list, x_list, age_design_list=age_list)

    print("Bootstrapping confidence intervals (n_boot=50, max_iter=10)...", flush=True)
    ci_res = model.bootstrap_ci(
        y_list,
        x_list,
        age_design_list=age_list,
        n_boot=50,
        ci=0.95,
        random_state=0,
        max_iter=10,
    )
    theta_lo, theta_hi = ci_res["theta_ci"]

    print("phi shape:", model.phi_.shape)
    print("theta shape:", model.theta_.shape)
    inv_drug_index = {v: k for k, v in drug_index.items()}
    for j in range(model.theta_.shape[0]):
        drug_name = inv_drug_index.get(j, f"drug_{j}")
        print(f"theta ({drug_name}, first 10 lags):", model.theta_[j, :10])
    if model.alpha_ is not None:
        print("age effects (alpha):", model.alpha_)

    # Plot log-RR and RR for each drug
    lags = np.arange(model.theta_.shape[1])
    lag_days = (lags + 1) * 7.0
    n_drugs = model.theta_.shape[0]

    fig, axes = plt.subplots(n_drugs, 2, figsize=(10, 3 * n_drugs), sharex=True)
    if n_drugs == 1:
        axes = np.array([axes])

    for j in range(n_drugs):
        drug_name = inv_drug_index.get(j, f"drug_{j}")
        log_rr = model.theta_[j]
        log_rr_lo = theta_lo[j]
        log_rr_hi = theta_hi[j]
        rr = np.exp(log_rr)
        rr_lo = np.exp(log_rr_lo)
        rr_hi = np.exp(log_rr_hi)

        ax_log = axes[j, 0]
        ax_rr = axes[j, 1]

        ax_log.plot(lag_days, log_rr, marker="o")
        ax_log.fill_between(lag_days, log_rr_lo, log_rr_hi, alpha=0.2)
        ax_log.set_title(f"{drug_name} log-RR")
        ax_log.set_ylabel("log(RR)")

        ax_rr.plot(lag_days, rr, marker="o")
        ax_rr.fill_between(lag_days, rr_lo, rr_hi, alpha=0.2)
        ax_rr.set_title(f"{drug_name} RR")
        ax_rr.set_ylabel("RR")

    for ax in axes[-1, :]:
        ax.set_xlabel("Days since exposure")

    fig.tight_layout()
    fig.savefig("convsccs_rr_plots.png", dpi=150)
    plt.close(fig)
    print("Saved plot to convsccs_rr_plots.png")


if __name__ == "__main__":
    main()
