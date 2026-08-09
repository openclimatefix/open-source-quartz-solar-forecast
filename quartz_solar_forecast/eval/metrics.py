import numpy as np
import pandas as pd


def bootstrap_mae_ci(
    error_df: pd.DataFrame, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
):
    """Percentile-bootstrap confidence interval for the MAE, resampling sites.

    Errors from the same PV site are correlated (same weather, same panel),
    so the independent unit for resampling is the site, not the row. Sites
    (pv_id) are resampled with replacement and the MAE recomputed for each
    resample.

    error_df needs columns:
    - pv_id
    - abs_error

    Returns (ci_low, ci_high); (nan, nan) if fewer than two sites.
    """
    site_errors = {
        pv_id: group["abs_error"].to_numpy() for pv_id, group in error_df.groupby("pv_id")
    }
    pv_ids = list(site_errors)
    if len(pv_ids) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_maes = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(len(pv_ids), size=len(pv_ids), replace=True)
        boot_maes[i] = np.concatenate([site_errors[pv_ids[j]] for j in chosen]).mean()
    return (
        float(np.quantile(boot_maes, alpha / 2)),
        float(np.quantile(boot_maes, 1 - alpha / 2)),
    )


def metrics(results_df: pd.DataFrame, pv_metadata: pd.DataFrame, include_night: bool = False):
    """
    Calculate and print metrics: MAE

    There is an option to include nighttime in the calculation of the MAE.

    results_df dataframe with the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - forecast_power
    - generation_power

    pv_metadata is a dataframe with the following columns
    - pv_id
    - capacity

    Returns a dataframe with one row per horizon group:
    - horizon_group
    - n (rows in the group)
    - mae
    - mae_ci_low / mae_ci_high (95% site-bootstrap interval)
    - mae_normalized
    """

    # remove night time
    if not include_night:
        results_df = results_df[results_df["generation_power"] > 0.1]

    # merge pv_metadata with results_df
    results_df = pd.merge(results_df, pv_metadata, on="pv_id")
    results_df = results_df.assign(
        abs_error=(results_df["forecast_power"] - results_df["generation_power"]).abs()
    )

    mae = np.round(results_df["abs_error"].mean(), 4)
    mae_normalized = np.round(
        ((results_df["forecast_power"] - results_df["generation_power"]) / results_df["capacity"])
        .abs()
        .mean(),
        4,
    )
    print(f"MAE: {mae} kw, normalized {100 * mae_normalized} %")

    # calculate metrics over the different horizons hours
    # find all unique horizon_hours
    horizon_hours = results_df["horizon_hour"].unique()
    horizon_groups = [[x, x] for x in horizon_hours]
    horizon_groups += [[3, 4], [5, 8], [9, 16], [17, 24], [24, 48], [0, 36]]

    rows = []
    for horizon_group in horizon_groups:
        horizon_group_df = results_df[
            results_df["horizon_hour"].between(horizon_group[0], horizon_group[1])
        ]
        if len(horizon_group_df) == 0:
            continue
        mae = np.round(horizon_group_df["abs_error"].mean(), 3)
        ci_low, ci_high = bootstrap_mae_ci(horizon_group_df)

        mae_normalized = np.round(
            (
                (horizon_group_df["forecast_power"] - horizon_group_df["generation_power"])
                / horizon_group_df["capacity"]
            )
            .abs()
            .mean(),
            3,
        )

        print(
            f"MAE for horizon {horizon_group}: {mae} "
            f"[{ci_low:.3g}, {ci_high:.3g}] (95% CI over sites, n={len(horizon_group_df)}). "
            f"mae_normalized: {100 * mae_normalized:.3g} %"
        )

        rows.append(
            {
                "horizon_group": f"{horizon_group[0]}-{horizon_group[1]}",
                "n": len(horizon_group_df),
                "mae": mae,
                "mae_ci_low": ci_low,
                "mae_ci_high": ci_high,
                "mae_normalized": mae_normalized,
            }
        )

    return pd.DataFrame(rows)
