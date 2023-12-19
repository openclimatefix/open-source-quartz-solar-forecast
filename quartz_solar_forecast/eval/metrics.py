import numpy as np
import pandas as pd


def metrics(results_df: pd.DataFrame):
    """
    Calculate and print metrics: MAE

    results_df dataframe with the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - forecast_power
    - generation_power

    """

    mae = np.round((results_df["forecast_power"] - results_df["generation_power"]).abs().mean(), 4)
    print(f"MAE: {mae}")

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
                / len(results_df_horizon) ** 0.5
            ),
            3,
        )

        print(f"MAE for horizon {horizon_hour}: {mae} +- {1.96*sem}")

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
                / len(horizon_group_df) ** 0.5
            ),
            3,
        )

        print(f"MAE for horizon {horizon_group}: {mae} +- {1.96*sem}")

        # TODO add more metrics using ocf_ml_metrics
