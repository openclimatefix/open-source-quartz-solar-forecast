import numpy as np
import pandas as pd
import pvlib
import pytest

from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.forecasts import v3
from quartz_solar_forecast.pydantic_models import PVSite

SITE = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=2.5, tilt=35, orientation=180)
TS = pd.Timestamp("2025-06-10 06:00")


def synthetic_weather(site: PVSite, start: pd.Timestamp, end: pd.Timestamp, *_args) -> pd.DataFrame:
    """Clear-sky hourly weather for every NWP model, in the Open-Meteo column layout."""
    times = pd.date_range(start, end, freq="h")
    clear = pvlib.location.Location(site.latitude, site.longitude).get_clearsky(
        times - pd.Timedelta("30min")
    )
    columns = {}
    for model in v3.NWP_MODELS:
        columns[f"shortwave_radiation_{model}"] = clear["ghi"].to_numpy()
        columns[f"direct_radiation_{model}"] = 0.7 * clear["ghi"].to_numpy()
        columns[f"cloud_cover_{model}"] = 10.0
        if model == v3.PRIMARY_MODEL:
            columns[f"temperature_2m_{model}"] = 15.0
            columns[f"wind_speed_10m_{model}"] = 10.0
            columns[f"precipitation_{model}"] = 0.0
    return pd.DataFrame(columns, index=times)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(v3, "fetch_hourly_weather", synthetic_weather)


def test_should_build_every_model_feature_half_hourly_when_given_weather():
    model = v3.load_v3_model()
    features = v3.make_features(
        SITE, synthetic_weather(SITE, TS.normalize(), TS + pd.Timedelta(days=3))
    )

    assert set(model["features"]) <= set(features.columns)
    assert features.index.freq == pd.Timedelta("30min")
    in_horizon = features.loc[TS : TS + pd.Timedelta(hours=v3.HORIZON_HOURS)]
    daytime = in_horizon[in_horizon["elevation"] > 10]
    assert daytime[model["features"]].drop(columns=["poa_ens_ratio_to_clear"]).notna().all().all()
    assert np.allclose(daytime["poa_spread"], 0)


def test_should_return_48_hours_every_15_minutes_from_ts(offline):
    forecast = v3.forecast_v3(SITE, TS)

    assert len(forecast) == 192
    assert forecast.index[0] == TS
    assert forecast.index.freq == pd.Timedelta("15min")
    assert forecast["power_kw"].notna().all()


def test_should_forecast_zero_at_night_and_positive_at_midday(offline):
    forecast = v3.forecast_v3(SITE, TS)["power_kw"]

    assert (forecast >= 0).all()
    assert forecast.at[pd.Timestamp("2025-06-11 01:00")] == 0
    assert (
        0.3 * SITE.capacity_kwp
        < forecast.at[pd.Timestamp("2025-06-11 12:00")]
        <= 1.1 * SITE.capacity_kwp
    )


def test_should_scale_with_capacity(offline):
    double = SITE.model_copy(update={"capacity_kwp": 2 * SITE.capacity_kwp})

    assert np.allclose(
        v3.forecast_v3(double, TS)["power_kw"], 2 * v3.forecast_v3(SITE, TS)["power_kw"]
    )


def test_should_dispatch_multi_nwp_from_run_forecast(offline):
    forecast = run_forecast(site=SITE, model="multi_nwp", ts=TS)

    assert len(forecast) == 192
