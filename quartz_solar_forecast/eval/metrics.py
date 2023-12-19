import numpy as np
import pandas as pd


def metrics(results_df: pd.DataFrame, pv_metadata: pd.DataFrame):
    """
    Calculate and print metrics: MAE

    results_df dataframe with the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - forecast_power
    - generation_power

    pv_metadata is a dataframe with the following columns
    - pv_id
    - capacity

    """

    # merge pv_metadata with results_df
    results_df = pd.merge(results_df, pv_metadata, on="pv_id")

    mae = np.round((results_df["forecast_power"] - results_df["generation_power"]).abs().mean(), 4)
    mae_normalized = np.round(
        ((results_df["forecast_power"] - results_df["generation_power"]) / results_df["capacity"])
        .abs()
        .mean(),
        4,
    )
    print(f"MAE: {mae} kw, normalized {mae_normalized} %")

    # calculate metrics over the different horizons hours
    # find all unique horizon_hours
    horizon_hours = results_df["horizon_hour"].unique()
    for horizon_hour in horizon_hours:
        # filter results_df to only include the horizon_hour
        results_df_horizon = results_df[results_df["horizon_hour"] == horizon_hour]
        mae = np.round(
            (results_df_horizon["forecast_power"] - results_df_horizon["generation_power"])
            .abs()
            .mean(),
            3,
        )
        sem = np.round(
            (
                (results_df_horizon["forecast_power"] - results_df_horizon["generation_power"])
                .abs()
                .std()
                / 50 ** 0.5
            ),
            3,
        )
        mae_normalized = np.round(
            (
                (results_df_horizon["forecast_power"] - results_df_horizon["generation_power"])
                / results_df_horizon["capacity"]
            )
            .abs()
            .mean(),
            3,
        )

        print(
            f"MAE for horizon {horizon_hour}: {mae} +- {1.96*sem}. Normalized MAE: {mae_normalized} %"
        )

    # calculate metrics over the different horizon groups
    horizon_groups = [[0, 0], [1, 1], [2, 2], [3, 4], [5, 8], [9, 16], [17, 24], [24, 48]]
    for horizon_group in horizon_groups:
        horizon_group_df = results_df[
            results_df["horizon_hour"].between(horizon_group[0], horizon_group[1])
        ]
        mae = np.round(
            (horizon_group_df["forecast_power"] - horizon_group_df["generation_power"])
            .abs()
            .mean(),
            3,
        )
        sem = np.round(
            (
                (horizon_group_df["forecast_power"] - horizon_group_df["generation_power"])
                .abs()
                .std()
                / 50 ** 0.5
            ),
            3,
        )

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
            f"MAE for horizon {horizon_group}: {mae} +- {1.96*sem}. mae_normalized: {mae_normalized} %"
        )

        # TODO add more metrics using ocf_ml_metrics
