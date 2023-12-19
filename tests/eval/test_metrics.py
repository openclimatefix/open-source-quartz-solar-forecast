from quartz_solar_forecast.eval.metrics import metrics
import pandas as pd
import numpy as np


def test_metrics():

    # create a fake dataframe

    results_df = pd.DataFrame(
        columns=[
            "pv_id",
            "timestamp",
            "horizon_hour",
            "forecast_power",
            "generation_power",
        ],
        data=np.random.random((100, 5)),
    )

    pv_metadata = pd.DataFrame(
        columns=[
            "pv_id",
            "capacity",
        ],
        data=np.random.random((100, 2)),
    )

    # call the metrics function
    metrics(results_df, pv_metadata)
