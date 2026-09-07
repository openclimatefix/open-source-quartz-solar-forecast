import numpy as np
import pandas as pd


def rmse(forecast, actual):
    """
    Root mean squared error (RMSE) between forecast and actual values.

    Accepts pandas Series or numpy arrays of the same length.
    Returns NaN for empty inputs.
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if forecast.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


def mbe(forecast, actual):
    """
    Mean bias error (MBE) between forecast and actual values.

    Positive values mean the forecast is on average too high,
    negative values mean it is on average too low.
    Accepts pandas Series or numpy arrays of the same length.
    Returns NaN for empty inputs.
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if forecast.size == 0:
        return float("nan")
    return float(np.mean(forecast - actual))


def rmse_normalized(forecast, actual, capacity):
    """
    Capacity-normalized RMSE: RMSE of errors divided by capacity.

    `capacity` may be a scalar or an array broadcastable to the inputs.
    Returns NaN for empty inputs.
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if forecast.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(((forecast - actual) / capacity) ** 2)))


def mbe_normalized(forecast, actual, capacity):
    """
    Capacity-normalized mean bias error: MBE of errors divided by capacity.

    `capacity` may be a scalar or an array broadcastable to the inputs.
    Returns NaN for empty inputs.
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if forecast.size == 0:
        return float("nan")
    return float(np.mean((forecast - actual) / capacity))


def metrics(results_df: pd.DataFrame, pv_metadata: pd.DataFrame, include_night: bool = False):
    """
    Calculate and print metrics: MAE, RMSE and MBE (with normalized versions)

    There is an option to include nighttime in the calculation of the metrics.

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

    # remove night time
    if not include_night:
        results_df = results_df[results_df["generation_power"] > 0.1]

    # merge pv_metadata with results_df
    results_df = pd.merge(results_df, pv_metadata, on="pv_id")

    error = results_df["forecast_power"] - results_df["generation_power"]

    mae = np.round(error.abs().mean(), 4)
    mae_normalized = np.round((error / results_df["capacity"]).abs().mean(), 4)
    rmse_value = np.round(rmse(results_df["forecast_power"], results_df["generation_power"]), 4)
    rmse_normalized_value = np.round(
        rmse_normalized(
            results_df["forecast_power"], results_df["generation_power"], results_df["capacity"]
        ),
        4,
    )
    mbe_value = np.round(mbe(results_df["forecast_power"], results_df["generation_power"]), 4)
    mbe_normalized_value = np.round(
        mbe_normalized(
            results_df["forecast_power"], results_df["generation_power"], results_df["capacity"]
        ),
        4,
    )

    print(f"MAE: {mae} kw, normalized {100 * mae_normalized} %")
    print(f"RMSE: {rmse_value} kw, normalized {100 * rmse_normalized_value} %")
    print(f"MBE: {mbe_value} kw, normalized {100 * mbe_normalized_value} %")

    # calculate metrics over the different horizons hours
    # find all unique horizon_hours
    horizon_hours = results_df["horizon_hour"].unique()
    horizon_groups = [[x, x] for x in horizon_hours]
    horizon_groups += [[3, 4], [5, 8], [9, 16], [17, 24], [24, 48], [0, 36]]

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
                / 50**0.5
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

        rmse_h = np.round(
            rmse(horizon_group_df["forecast_power"], horizon_group_df["generation_power"]), 3
        )
        mbe_h = np.round(
            mbe(horizon_group_df["forecast_power"], horizon_group_df["generation_power"]), 3
        )

        print(
            f"MAE for horizon {horizon_group}: {mae} +- {1.96 * sem:.3g}. "
            f"mae_normalized: {100 * mae_normalized:.3g} %. "
            f"rmse: {rmse_h:.3g} kw. mbe: {mbe_h:.3g} kw"
        )
