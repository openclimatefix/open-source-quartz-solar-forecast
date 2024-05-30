from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime, timedelta

def test_run_forecast():
    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    ts = datetime.today() - timedelta(weeks=2)

    # run model with icon and gfs nwp
    predications_df_gfs = run_forecast(site=site, model="gb", ts=ts, nwp_source="gfs")
    predications_df_icon = run_forecast(site=site, model="gb", ts=ts, nwp_source="icon")
    predications_df_xgb = run_forecast(site=site, ts=ts)

    print("\n Prediction based on GFS NWP\n")
    print(predications_df_gfs)
    print(f" Max: {predications_df_gfs['power_wh'].max()}")

    print("\n Prediction based on ICON NWP\n")
    print(predications_df_icon)
    print(f" Max: {predications_df_icon['power_wh'].max()}")

    print("\n Prediction based on XGB\n")
    print(predications_df_xgb)
    print(f" Max: {predications_df_xgb['power_wh'].max()}")


def test_run_forecast_historical():

    # model input data creation
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    ts = datetime.today() - timedelta(days=200)

    # run model with icon and gfs nwp
    predications_df_gfs = run_forecast(site=site, ts=ts, model="gb", nwp_source="gfs")
    predications_df_icon = run_forecast(site=site, ts=ts, model="gb", nwp_source="icon")
    predications_df_xgb = run_forecast(site=site, ts=ts, model="xgb")

    print("\nPrediction for a date more than 180 days in the past")

    print("\n Prediction based on GFS NWP\n")
    print(predications_df_gfs)
    print(f" Max: {predications_df_gfs['power_wh'].max()}")

    print("\n Prediction based on ICON NWP\n")
    print(predications_df_icon)
    print(f" Max: {predications_df_icon['power_wh'].max()}")
    
    print("\n Prediction based on XGB\n")
    print(predications_df_xgb)

