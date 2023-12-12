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

    returns a dataframe with the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - forecast_power
    - generation_power

    """

    # rename power to forecast_power
    forecast_df = forecast_df.rename(columns={"power_wh": "forecast_power"})

    # rename power to ground_truth_power
    ground_truth_df = ground_truth_df.rename(columns={"value": "generation_power"})

    # make pv_ids are ints
    forecast_df["pv_id"] = forecast_df["pv_id"].astype(int)
    ground_truth_df["pv_id"] = ground_truth_df["pv_id"].astype(int)

    # merge the two dataframes
    print(forecast_df)
    print(ground_truth_df)
    combined_df = pd.merge(forecast_df, ground_truth_df, on=["timestamp", "pv_id", "horizon_hour"])

    return combined_df

