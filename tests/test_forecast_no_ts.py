import pandas as pd
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite


def test_run_forecast_no_ts():
    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

    current_ts = pd.Timestamp.now().round("15min")

    # run ocf model with no ts
    predications_df = run_forecast(site=site, model="ocf")
    # check current ts agrees with dataset
    assert predications_df.index.min() == current_ts

    print(predications_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predications_df['power_wh'].max()}")

    current_ts = pd.Timestamp.now().floor("15min")
    # run tryolabs model with no ts
    predications_df = run_forecast(site=site)
    # check current ts agrees with dataset
    assert predications_df.index.min() == current_ts

    print(predications_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predications_df['power_wh'].max()}")