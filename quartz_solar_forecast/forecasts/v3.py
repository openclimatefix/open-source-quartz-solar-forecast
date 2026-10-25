"""
v3 model: gradient boosting on three NWP models (ICON, ECMWF IFS, GFS).

Each model's irradiance is converted to plane-of-array irradiance for the site's tilt and
orientation. The trees get all three, their spread, and the irradiance an hour either side.
Doesn't use live generation data.

Trained with scripts/v3/train.py on uk_pv sites, using the Open-Meteo Previous Runs API (the run
issued 24-48 h before each time). Live forecasts come from newer runs than that.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pvlib
import requests
from retry_requests import retry

from quartz_solar_forecast.pydantic_models import PVSite

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = f"{dir_path}/../models/v3-0.1.0.pkl"

NWP_MODELS = ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"]
PRIMARY_MODEL = "icon_seamless"
# the primary model also supplies the weather features; the others only irradiance and cloud
PRIMARY_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover",
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
]
OTHER_VARIABLES = ["shortwave_radiation", "direct_radiation", "cloud_cover"]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# the forecast API serves roughly the last three months; older dates come from the archive
FORECAST_API_DAYS_BACK = 90
# below this cos(zenith) the direct-normal conversion blows up (sun on the horizon)
MIN_COS_ZENITH = 0.05
# clear-sky POA below this (kW/kWp) makes the ratio-to-clear feature meaningless
MIN_CLEAR_POA = 0.02
HORIZON_HOURS = 48


def fetch_hourly_weather(
    site: PVSite, start: pd.Timestamp, end: pd.Timestamp, now: datetime
) -> pd.DataFrame:
    """Hourly forecasts of every NWP model, columns named "<variable>_<model>", naive GMT index."""
    url = FORECAST_URL if (now - start).days <= FORECAST_API_DAYS_BACK else HISTORICAL_FORECAST_URL
    session = retry(requests.Session(), retries=5, backoff_factor=0.2)
    frames = []
    for model in NWP_MODELS:
        variables = PRIMARY_VARIABLES if model == PRIMARY_MODEL else OTHER_VARIABLES
        params = {
            "latitude": site.latitude,
            "longitude": site.longitude,
            "start_date": f"{start.date()}",
            "end_date": f"{end.date()}",
            "hourly": ",".join(variables),
            "models": model,
            "timezone": "GMT",
        }
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        hourly = pd.DataFrame(response.json()["hourly"])
        hourly = hourly.set_index(pd.to_datetime(hourly.pop("time")))
        frames.append(hourly.add_suffix(f"_{model}"))
    return pd.concat(frames, axis=1).astype("float64")


def split_irradiance(
    ghi: np.ndarray, direct_horizontal: np.ndarray, zenith: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Direct-normal and diffuse-horizontal irradiance from global and direct horizontal."""
    cos_zenith = np.cos(np.radians(zenith))
    dni = np.where(
        cos_zenith > MIN_COS_ZENITH, direct_horizontal / np.maximum(cos_zenith, MIN_COS_ZENITH), 0.0
    )
    return dni, np.clip(ghi - direct_horizontal, 0, None)


def plane_of_array(
    site: PVSite, ghi: np.ndarray, direct_horizontal: np.ndarray, solar: pd.DataFrame
) -> np.ndarray:
    """Isotropic plane-of-array irradiance in kW per kWp."""
    zenith = solar["apparent_zenith"].to_numpy()
    dni, dhi = split_irradiance(ghi, direct_horizontal, zenith)
    poa = pvlib.irradiance.get_total_irradiance(
        site.tilt,
        site.orientation,
        zenith,
        solar["azimuth"].to_numpy(),
        dni,
        ghi,
        dhi,
        model="isotropic",
    )["poa_global"]
    return np.nan_to_num(np.asarray(poa, dtype=float)) / 1000


def make_features(site: PVSite, hourly: pd.DataFrame) -> pd.DataFrame:
    """Half-hourly model features, indexed by period end (naive GMT).

    Open-Meteo radiation is the mean over the preceding hour, so the half-hour ending at T
    takes the hourly value labelled at T rounded up to the hour.
    """
    times = pd.date_range(hourly.index[0], hourly.index[-1], freq="30min")
    weather = hourly.reindex(times.ceil("h")).set_axis(times)
    midpoints = times - pd.Timedelta("15min")
    location = pvlib.location.Location(site.latitude, site.longitude)
    solar = location.get_solarposition(midpoints)
    clearsky = location.get_clearsky(midpoints, model="ineichen")
    zenith = solar["apparent_zenith"].to_numpy()

    primary = {v: weather[f"{v}_{PRIMARY_MODEL}"].to_numpy() for v in PRIMARY_VARIABLES}
    ghi, direct_horizontal = primary["shortwave_radiation"], primary["direct_radiation"]
    dni, dhi = split_irradiance(ghi, direct_horizontal, zenith)
    clear_poa = pvlib.irradiance.get_total_irradiance(
        site.tilt,
        site.orientation,
        zenith,
        solar["azimuth"].to_numpy(),
        clearsky["dni"].to_numpy(),
        clearsky["ghi"].to_numpy(),
        clearsky["dhi"].to_numpy(),
        model="isotropic",
    )["poa_global"]
    doy = times.dayofyear.to_numpy()
    features = pd.DataFrame(
        {
            "poa_kw_per_kwp": plane_of_array(site, ghi, direct_horizontal, solar),
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "clearsky_ghi": clearsky["ghi"].to_numpy(),
            "clearsky_index": np.where(
                clearsky["ghi"] > 20, ghi / clearsky["ghi"].clip(lower=1), np.nan
            ),
            "poa_clearsky": np.nan_to_num(np.asarray(clear_poa, dtype=float)) / 1000,
            "elevation": 90 - zenith,
            "azimuth": solar["azimuth"].to_numpy(),
            "cloud_cover": primary["cloud_cover"],
            "temperature_2m": primary["temperature_2m"],
            "wind_speed_10m": primary["wind_speed_10m"],
            "precipitation": primary["precipitation"],
            "tilt": site.tilt,
            "orientation": site.orientation,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=times,
    )
    for model in NWP_MODELS[1:]:
        features[f"{model}_poa"] = plane_of_array(
            site,
            weather[f"shortwave_radiation_{model}"].to_numpy(),
            weather[f"direct_radiation_{model}"].to_numpy(),
            solar,
        )
        features[f"{model}_ghi"] = weather[f"shortwave_radiation_{model}"].to_numpy()
        features[f"{model}_cloud"] = weather[f"cloud_cover_{model}"].to_numpy()
    all_poa = features[["poa_kw_per_kwp"] + [f"{m}_poa" for m in NWP_MODELS[1:]]]
    features["poa_ensemble"] = all_poa.mean(axis=1)
    features["poa_spread"] = all_poa.std(axis=1)
    return add_temporal_context(features)


def add_temporal_context(features: pd.DataFrame) -> pd.DataFrame:
    """NWP cloud timing is often off by an hour or two, so neighbouring hours carry signal."""
    poa, clear = features["poa_ensemble"], features["poa_clearsky"]
    day = features.index.normalize()
    features["poa_ens_lag1h"] = poa.shift(2)
    features["poa_ens_lead1h"] = poa.shift(-2)
    features["poa_ens_mean3h"] = poa.rolling(7, center=True, min_periods=1).mean()
    features["poa_ens_std3h"] = poa.rolling(7, center=True, min_periods=1).std()
    features["day_clearsky_index"] = poa.groupby(day).transform("sum") / clear.groupby(
        day
    ).transform("sum").clip(lower=1e-3)
    features["poa_ens_ratio_to_clear"] = poa / clear.where(clear > MIN_CLEAR_POA)
    return features


def load_v3_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast_v3(site: PVSite, ts: pd.Timestamp, model: dict | None = None) -> pd.DataFrame:
    """Forecast power (kW) every 15 minutes for 48 hours from ts."""
    if model is None:
        model = load_v3_model()
    ts = pd.Timestamp(ts).floor("15min")
    # whole days either side so the daily and rolling context features see complete days
    hourly = fetch_hourly_weather(
        site, ts.normalize(), ts + pd.Timedelta(hours=HORIZON_HOURS + 24), datetime.now()
    )
    features = make_features(site, hourly)

    kw_per_kwp = np.clip(model["model"].predict(features[model["features"]]), 0, None)
    kw_per_kwp[features["elevation"].to_numpy() <= 0] = 0.0
    # each value is the mean over the half-hour ending at its index; place it at the midpoint
    half_hourly = pd.Series(
        kw_per_kwp * site.capacity_kwp, index=features.index - pd.Timedelta("15min")
    )
    times = pd.date_range(start=ts, periods=HORIZON_HOURS * 4, freq="15min")
    power = (
        half_hourly.reindex(half_hourly.index.union(times))
        .interpolate(method="time")
        .reindex(times)
    )
    return pd.DataFrame({"power_kw": power.to_numpy()}, index=times)
