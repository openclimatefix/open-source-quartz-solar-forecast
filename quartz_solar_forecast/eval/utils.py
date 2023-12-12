import pandas as pd





def combine_forecast_ground_truth(forecast_df, ground_truth_df):
    """
    Combine the forecast results with the ground truth (ts, id, horizon (in hours), pred, truth, diff)


    forecast_df should have the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - power

    ground_truth_df should have the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - power

    """

    # rename power to forecast_power
    forecast_df = forecast_df.rename(columns={"power": "forecast_power"})

    # rename power to ground_truth_power
    ground_truth_df = ground_truth_df.rename(columns={"power": "ground_truth_power"})

    # merge the two dataframes
    combined_df = pd.merge(forecast_df, ground_truth_df, on=["timestamp", "pv_id", "horizon_hours"])

    return combined_df

