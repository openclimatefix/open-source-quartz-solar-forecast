from quartz_solar_forecast.eval.forecast import run_forecast

import pandas as pd


def test_run_forecast():
    pv_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2021-01-26 01:15:00"),
                "latitude": 51.5,
                "longitude": 0.0,
                "capacity": 1.0,
                "pv_id": 1.0,
            }
        ]
    )

    nwp_df = pd.DataFrame(
        columns=[
            "pv_id",
            "timestamp",
            "t",
            "prate",
            "dswrf",
            "dlwrf",
            "lcc",
            "mcc",
            "hcc",
            "vis",
            "si10",
            "time",
        ],
        data=[
            [
                1,
                pd.Timestamp("2021-01-26 01:15:00"),
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                pd.Timestamp("2021-01-26 01:15:00"),
            ],
            [
                1,
                pd.Timestamp("2021-01-26 01:15:00"),
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                pd.Timestamp("2021-01-26 01:30:00"),
            ],
        ],
    )

    # format timestamp
    pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"])
    nwp_df["timestamp"] = pd.to_datetime(nwp_df["timestamp"])
    nwp_df["time"] = pd.to_datetime(nwp_df["time"])

    _ = run_forecast(pv_df=pv_df, nwp_df=nwp_df, nwp_source="ICON")
