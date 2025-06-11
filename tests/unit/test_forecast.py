from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from tests.unit.mocks import mock_weather_api
from datetime import datetime, timedelta

import numpy as np
import pytest


# Shared fixture for site input
@pytest.fixture
def site():
    return PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)


def test_run_forecast_invalid_model(site):
    with pytest.raises(ValueError, match="Unsupported model:"):
        run_forecast(site=site, model="invalid_model")


def test_run_forecast_invalid_nwp_source(site):
    with pytest.raises(Exception, match="Source"):
        run_forecast(site=site, model="gb", ts=None, nwp_source="invalid_nwp_source")


def test_run_forecast(site, mock_weather_api):
    ts = datetime.today() - timedelta(weeks=2)

    # run model with icon, gfs and ukmo nwp
    predications_df_gfs = run_forecast(site=site, model="gb", ts=ts, nwp_source="gfs")
    predications_df_icon = run_forecast(site=site, model="gb", ts=ts, nwp_source="icon")
    predications_df_ukmo = run_forecast(site=site, model="gb", ts=ts, nwp_source="ukmo")
    predications_df_xgb = run_forecast(site=site, ts=ts)

    print("\n Prediction based on GFS NWP\n")
    print(predications_df_gfs)
    print(f" Max: {predications_df_gfs['power_kw'].max()}")

    print("\n Prediction based on ICON NWP\n")
    print(predications_df_icon)
    print(f" Max: {predications_df_icon['power_kw'].max()}")

    print("\n Prediction based on UKMO NWP\n")
    print(predications_df_ukmo)
    print(f" Max: {predications_df_ukmo['power_kw'].max()}")

    print("\n Prediction based on XGB\n")
    print(predications_df_xgb)
    print(f" Max: {predications_df_xgb['power_kw'].max()}")


def test_run_forecast_historical(site, mock_weather_api):
    ts = datetime.today() - timedelta(days=200)

    # run model with icon, gfs and ukmo nwp
    predications_df_gfs = run_forecast(site=site, ts=ts, model="gb", nwp_source="gfs")
    predications_df_icon = run_forecast(site=site, ts=ts, model="gb", nwp_source="icon")
    predications_df_ukmo = run_forecast(site=site, ts=ts, model="gb", nwp_source="ukmo")
    predications_df_xgb = run_forecast(site=site, ts=ts, model="xgb")

    print("\nPrediction for a date more than 180 days in the past")

    print("\n Prediction based on GFS NWP\n")
    print(predications_df_gfs)
    print(f" Max: {predications_df_gfs['power_kw'].max()}")

    print("\n Prediction based on ICON NWP\n")
    print(predications_df_icon)
    print(f" Max: {predications_df_icon['power_kw'].max()}")

    print("\n Prediction based on UKMO NWP\n")
    print(predications_df_ukmo)
    print(f" Max: {predications_df_ukmo['power_kw'].max()}")

    print("\n Prediction based on XGB\n")
    print(predications_df_xgb)


def test_large_capacity(mock_weather_api):
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=4)
    site_large = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=4000)
    ts = datetime.today() - timedelta(weeks=2)

    # run model with icon, gfs and ukmo nwp
    predications_df = run_forecast(site=site, model="gb", ts=ts, nwp_source="gfs")
    predications_df_large = run_forecast(site=site_large, model="gb", ts=ts, nwp_source="gfs")

    assert np.round(predications_df["power_kw"].sum() * 1000, 8) == np.round(
        predications_df_large["power_kw"].sum(), 8
    )
