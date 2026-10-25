"""
Backtest the shipped gb model and two naive baselines on the sites in sites.csv.

gb runs exactly as forecast.py does (psp model-0.4.0, ICON from Open-Meteo, GFS visibility,
no live PV), once with the inputs as on main and once with the unit fixes from PR #376.
Day-ahead cloud layers and visibility aren't archived, so gb gets Open-Meteo's Historical
Forecast API (stitched short-lead runs): better weather than it would have live.

Writes <out>/gb.parquet: one row per site, init and half-hour (period end, naive GMT).

Usage (see README.md):
    python scripts/v3/backtest_gb.py --uk-pv-dir <uk_pv> --cache-dir <dir> --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import DATA_END, DATA_START, FIRST_INIT, LAST_INIT, get_json, load_sites, load_truth
from common import pv_site as make_pv_site
from psp.serialization import load_model

from quartz_solar_forecast.data import format_nwp_data, make_pv_data
from quartz_solar_forecast.forecasts import forecast_v1_tilt_orientation
from quartz_solar_forecast.forecasts.v1_tilt_orientation import dir_path

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ICON_VARIABLES = [
    "temperature_2m", "precipitation", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation", "direct_radiation", "dew_point_2m",
]  # fmt: skip
GB_FEATURE_ORDER = ["t", "prate", "lcc", "mcc", "hcc", "si10", "dswrf", "dlwrf", "vis"]
STEFAN_BOLTZMANN = 5.670374e-8  # W m-2 K-4


def short_lead_weather(site: pd.Series, cache_dir: Path) -> pd.DataFrame:
    """Hourly ICON plus GFS visibility, as get_nwp fetches them, cached as JSON."""
    cache_file = cache_dir / "historical_forecast" / f"{site['ss_id']}.json"
    if not cache_file.exists():
        params = {
            "latitude": site["latitude_rounded"],
            "longitude": site["longitude_rounded"],
            "start_date": f"{DATA_START:%Y-%m-%d}",
            "end_date": f"{DATA_END:%Y-%m-%d}",
            "timezone": "GMT",
        }
        icon = get_json(
            HISTORICAL_FORECAST_URL,
            params | {"hourly": ",".join(ICON_VARIABLES), "models": "icon_seamless"},
        )["hourly"]
        gfs = get_json(
            HISTORICAL_FORECAST_URL, params | {"hourly": "visibility", "models": "gfs_seamless"}
        )["hourly"]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"icon": icon, "gfs": gfs}))
    cached = json.loads(cache_file.read_text())
    df = pd.DataFrame(cached["icon"]).assign(visibility=cached["gfs"]["visibility"])
    return df.set_index(pd.to_datetime(df.pop("time")))


def downward_longwave(temperature_c, dew_point_c, cloud_cover_pct):
    """Brutsaert (1975) clear-sky emissivity with a (1 + 0.22 c^2) cloud factor, as in PR #376."""
    temperature_k = temperature_c + 273.15
    vapour_pressure_hpa = 6.112 * np.exp(17.67 * dew_point_c / (dew_point_c + 243.5))
    emissivity = 1.24 * (vapour_pressure_hpa / temperature_k) ** (1 / 7)
    cloud = np.clip(cloud_cover_pct / 100.0, 0, 1)
    return STEFAN_BOLTZMANN * temperature_k**4 * emissivity * (1 + 0.22 * cloud**2)


def gb_inputs(weather: pd.DataFrame, pr376: bool) -> pd.DataFrame:
    """main passes Open-Meteo units straight through (and direct radiation as dlwrf);
    PR #376 converts to the units the model was trained on and estimates longwave."""
    w = weather
    if pr376:
        inputs = {
            "t": w["temperature_2m"] + 273.15,
            "prate": w["precipitation"] / 3600.0,
            "si10": w["wind_speed_10m"] / 3.6,
            "dlwrf": downward_longwave(w["temperature_2m"], w["dew_point_2m"], w["cloud_cover"]),
        }
    else:
        inputs = {
            "t": w["temperature_2m"],
            "prate": w["precipitation"],
            "si10": w["wind_speed_10m"],
            "dlwrf": w["direct_radiation"],
        }
    inputs |= {
        "lcc": w["cloud_cover_low"],
        "mcc": w["cloud_cover_mid"],
        "hcc": w["cloud_cover_high"],
        "dswrf": w["shortwave_radiation"],
        "vis": w["visibility"],
    }
    return pd.DataFrame(inputs, index=w.index)[GB_FEATURE_ORDER].astype("float64")


def to_half_hourly(power_15min: pd.Series) -> pd.Series:
    """Average 15-minute points into half-hours labelled by period end (complete ones only)."""
    resampled = power_15min.resample("30min", label="right", closed="left")
    return resampled.mean()[resampled.count() == 2]


def gb_forecasts(site: pd.Series, weather: pd.DataFrame, model, inits) -> pd.DataFrame:
    pv_site = make_pv_site(site)
    inputs = {"gb": gb_inputs(weather, pr376=False), "gb_pr376": gb_inputs(weather, pr376=True)}
    frames = []
    for ts in inits:
        columns = {}
        for name, df in inputs.items():
            # forecast.py fetches 7 days of weather starting at midnight of the init day
            window = df.loc[ts.normalize() : ts.normalize() + pd.Timedelta(days=7)]
            nwp_xr = format_nwp_data(window, "icon", pv_site)
            preds = forecast_v1_tilt_orientation(
                "icon", nwp_xr, make_pv_data(site=pv_site, ts=ts), ts, model=model
            )
            columns[name] = to_half_hourly(preds["power_kw"])
        frame = pd.DataFrame(columns)
        frames.append(frame[frame.index <= ts + pd.Timedelta(hours=48)].assign(init=ts))
    return pd.concat(frames)


def baselines(truth: pd.Series, ts: pd.Timestamp, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """persistence: same time on the last full day before ts.
    climatology: same time, averaged over the 7 days before ts."""
    time_of_day = targets - targets.normalize()
    day_before = ts.normalize() - pd.Timedelta(days=1)
    week = np.stack(
        [truth.reindex(day_before - pd.Timedelta(days=d) + time_of_day) for d in range(7)]
    )
    with np.errstate(all="ignore"):
        climatology = np.nanmean(week, axis=0)
    return pd.DataFrame({"persistence": week[0], "climatology": climatology}, index=targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uk-pv-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sites = load_sites(args.uk_pv_dir)
    truth = load_truth(args.uk_pv_dir, sites)
    model = load_model(f"{dir_path}/../models/model-0.4.0.pkl")
    inits = pd.date_range(FIRST_INIT, LAST_INIT, freq="D")

    parts = []
    for i, (_, site) in enumerate(sites.iterrows(), start=1):
        forecasts = gb_forecasts(site, short_lead_weather(site, args.cache_dir), model, inits)
        site_truth = truth.loc[site["ss_id"]]
        for ts, group in forecasts.groupby("init"):
            parts.append(
                group.join(baselines(site_truth, ts, group.index)).assign(ss_id=site["ss_id"])
            )
        print(f"[backtest_gb] site {i}/{len(sites)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    out = pd.concat(parts).rename_axis("timestamp").reset_index()
    out.to_parquet(args.out / "gb.parquet")
    print(f"[backtest_gb] wrote {args.out / 'gb.parquet'} ({len(out):,} rows)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
