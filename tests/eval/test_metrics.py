from quartz_solar_forecast.eval.metrics import metrics
import pandas as pd
import numpy as np


def test_metrics():

    # create a fake dataframe

    results_df = pd.DataFrame(
        columns=[
            "id",
            "timestamp",
            "horizon_hours",
            "forecast_power",
            "generation_power",
        ], data=np.random.random((100,5)))

    # call the metrics function
    metrics(results_df)