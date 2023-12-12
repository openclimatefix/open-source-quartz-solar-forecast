"""
To evaluate the performance of the solar forecast, a predefined testset is used.

A file has been added to this branch (make-testset) which defines a set of random timestamps and sites ids. 
This contains 50 sites each with 50 timestamps to make 2500 samples in total.

"""
from quartz_solar_forecast.eval.nwp import get_nwp
from quartz_solar_forecast.eval.forecast import run_forecast
from quartz_solar_forecast.eval.utils import combine_forecast_ground_truth

import pandas as pd


def run_eval(testset_path):
    # load testset from csv
    testset = pd.read_csv(testset_path)

    # Extract generation data and metadata for specific sites and timestamps for the testset from Hugging Face. (Zak)

    # Split data into PV inputs and ground truth. (Zak)
    ground_truth_df = None # TODO

    # Collect NWP data from Hugging Face, ICON. (Peter)
    nwp_df = get_nwp(testset)

    # Run forecast with PV and NWP inputs.
    # TODO updatepv_df
    predictions_df = run_forecast(pv_df=None, nwp_df=nwp_df)

    # Combine the forecast results with the ground truth (ts, id, horizon (in hours), pred, truth, diff)
    results_df = combine_forecast_ground_truth(predictions_df, ground_truth_df)

    # Save file
    results_df.to_csv("results.csv")

    # Calculate and print metrics: MAE

    # Visulisations
