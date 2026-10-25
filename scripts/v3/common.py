"""Shared data loading for training and backtesting the v3 (multi_nwp) model."""

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from quartz_solar_forecast.forecasts.v3 import (
    NWP_MODELS,
    OTHER_VARIABLES,
    PRIMARY_MODEL,
    PRIMARY_VARIABLES,
    make_features,
)
from quartz_solar_forecast.pydantic_models import PVSite

SITES_FILE = Path(__file__).parent / "sites.csv"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# forecasts are issued daily at 00:00 from FIRST_INIT to LAST_INIT; data has a margin either side
FIRST_INIT, LAST_INIT = pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-30")
DATA_START, DATA_END = pd.Timestamp("2024-03-24"), pd.Timestamp("2025-04-08")
# uk_pv README: readings above kWp x 750 Wh per half hour are not physical
MAX_WH_PER_KWP = 750
MAX_RATE_LIMIT_WAITS = 40

FEATURES = [
    "poa_kw_per_kwp", "ghi", "dni", "dhi", "clearsky_ghi", "clearsky_index", "poa_clearsky",
    "elevation", "azimuth", "cloud_cover", "temperature_2m", "wind_speed_10m", "precipitation",
    "tilt", "orientation", "doy_sin", "doy_cos",
    *[f"{m}_{v}" for m in NWP_MODELS[1:] for v in ("poa", "ghi", "cloud")],
    "poa_ensemble", "poa_spread",
    "poa_ens_lag1h", "poa_ens_lead1h", "poa_ens_mean3h", "poa_ens_std3h",
    "day_clearsky_index", "poa_ens_ratio_to_clear",
]  # fmt: skip


def gradient_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=0,
    )


def load_sites(uk_pv_dir: Path) -> pd.DataFrame:
    """sites.csv joined to uk_pv metadata (ss_id, split, location, tilt, orientation, kWp)."""
    meta = pd.read_csv(uk_pv_dir / "metadata.csv")
    return pd.read_csv(SITES_FILE).merge(meta, on="ss_id", validate="1:1")


def pv_site(site: pd.Series) -> PVSite:
    return PVSite(
        latitude=site["latitude_rounded"],
        longitude=site["longitude_rounded"],
        capacity_kwp=site["kWp"],
        tilt=site["tilt"],
        orientation=site["orientation"],
    )


def load_truth(uk_pv_dir: Path, sites: pd.DataFrame) -> pd.Series:
    """Mean power (kW) over each half-hour, indexed by (ss_id, naive GMT period end), cleaned
    as the uk_pv README advises."""
    table = ds.dataset(uk_pv_dir / "30_minutely", format="parquet", partitioning="hive").to_table(
        columns=["ss_id", "datetime_GMT", "generation_Wh"],
        filter=ds.field("ss_id").isin(sites["ss_id"].tolist())
        & (ds.field("datetime_GMT") >= DATA_START.tz_localize("UTC"))
        & (ds.field("datetime_GMT") < DATA_END.tz_localize("UTC")),
    )
    df = table.to_pandas()
    kwp = df["ss_id"].map(sites.set_index("ss_id")["kWp"])
    df = df[(df["generation_Wh"] >= 0) & (df["generation_Wh"] <= kwp * MAX_WH_PER_KWP)]
    bad = pd.read_csv(
        uk_pv_dir / "bad_data.csv", parse_dates=["start_datetime_GMT", "end_datetime_GMT"]
    )
    for period in bad[bad["ss_id"].isin(sites["ss_id"])].itertuples():
        end = period.end_datetime_GMT
        if pd.isna(end):
            end = df["datetime_GMT"].max()
        in_period = df["datetime_GMT"].between(period.start_datetime_GMT, end)
        df = df[~((df["ss_id"] == period.ss_id) & in_period)]
    df["timestamp"] = df["datetime_GMT"].dt.tz_convert(None)
    df["power_kw"] = df["generation_Wh"] * 2 / 1000
    return df.set_index(["ss_id", "timestamp"])["power_kw"].sort_index()


def get_json(url: str, params: dict) -> dict:
    """GET with waits for Open-Meteo's rate limit (the quota resets on the hour)."""
    for _ in range(MAX_RATE_LIMIT_WAITS):
        try:
            response = requests.get(url, params=params, timeout=120)
        except requests.exceptions.Timeout:
            time.sleep(10)
            continue
        if response.status_code == 200 and not response.content.strip():
            time.sleep(10)
            continue
        if response.status_code != 429:
            break
        time.sleep(60)
    response.raise_for_status()
    return response.json()


def previous_runs(site: pd.Series, model: str, cache_dir: Path) -> pd.DataFrame:
    """Day-1 and day-2 forecasts ("<variable>_previous_day<n>") for one model, cached as JSON."""
    variables = PRIMARY_VARIABLES if model == PRIMARY_MODEL else OTHER_VARIABLES
    cache_file = cache_dir / "previous_runs" / model / f"{site['ss_id']}.json"
    if not cache_file.exists():
        params = {
            "latitude": site["latitude_rounded"],
            "longitude": site["longitude_rounded"],
            "start_date": f"{DATA_START:%Y-%m-%d}",
            "end_date": f"{DATA_END:%Y-%m-%d}",
            "hourly": ",".join(f"{v}_previous_day{d}" for v in variables for d in (1, 2)),
            "models": model,
            "timezone": "GMT",
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(get_json(PREVIOUS_RUNS_URL, params)["hourly"]))
    hourly = pd.DataFrame(json.loads(cache_file.read_text()))
    return hourly.set_index(pd.to_datetime(hourly.pop("time")))


def day_ahead_features(site: pd.Series, lead_day: int, cache_dir: Path) -> pd.DataFrame:
    """v3 features from the runs issued lead_day days before each time (1: 24-48 h lead)."""
    suffix = f"_previous_day{lead_day}"
    hourly = []
    for model in NWP_MODELS:
        runs = previous_runs(site, model, cache_dir)
        runs = runs[[c for c in runs.columns if c.endswith(suffix)]]
        hourly.append(runs.rename(columns=lambda c: f"{c.removesuffix(suffix)}_{model}"))
    return make_features(pv_site(site), pd.concat(hourly, axis=1))
