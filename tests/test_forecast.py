from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite


def test_run_forecast():
    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

    # run model with icon and gfs nwp
    predications_df_gfs = run_forecast(site=site, ts='2023-12-30', nwp_source="gfs")
    predications_df_icon = run_forecast(site=site, ts='2023-12-30', nwp_source="icon")

    print("\nPrediction based on GFS NWP\n")
    print(predications_df_gfs)
    print(f"Max: {predications_df_gfs['power_wh'].max()}")

    print("\nPrediction based on ICON NWP\n")
    print(predications_df_icon)
    print(f"Max: {predications_df_icon['power_wh'].max()}")
