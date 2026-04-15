import os

import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import HfFileSystem

fs = HfFileSystem()


def get_pv_metadata(testset: pd.DataFrame):
    # download from hugginface or load from cache
    cache_dir = "data/pv"
    metadata_file = f"{cache_dir}/metadata.csv"
    if not os.path.exists(metadata_file):
        os.makedirs(cache_dir, exist_ok=True)
        fs.get("datasets/openclimatefix/uk_pv/metadata.csv", metadata_file)

    # Load in the dataset
    metadata_df = pd.read_csv(metadata_file)

    # join metadata with testset
    metadata_df = metadata_df.rename(columns={"ss_id": "pv_id"})
    combined_data = testset.merge(metadata_df, on="pv_id", how="left")

    # only keep the columns we need
    combined_data = combined_data[
        ["pv_id", "timestamp", "latitude_rounded", "longitude_rounded", "kwp"]
    ]

    # rename latitude_rounded to latitude and longitude_rounded to longitude
    combined_data = combined_data.rename(
        columns={
            "latitude_rounded": "latitude",
            "longitude_rounded": "longitude",
            "kwp": "capacity",
        }
    )

    # format datetime
    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])

    return combined_data


def get_pv_truth(testset: pd.DataFrame):
    """Extract PV ground-truth values for all test entries and forecast horizons.

    Vectorized implementation: expands all (pv_id, timestamp, horizon) combinations
    in one step, then batch-selects from the xarray dataset. Avoids per-row iterrows()
    and per-horizon xarray.sel() calls.

    Closes #344.
    """
    print("Loading PV data")

    # download from huggingface or load from cache
    cache_dir = "data/pv"
    metadata_file = f"{cache_dir}/pv.netcdf"
    if not os.path.exists(metadata_file):
        print("Loading from HF)")
        os.makedirs(cache_dir, exist_ok=True)
        fs.get("datasets/openclimatefix/uk_pv/pv.netcdf", metadata_file)

    # Load in the dataset
    pv_ds = xr.open_dataset(metadata_file, engine="h5netcdf")

    # Build all (pv_id, base_timestamp, horizon) combinations at once
    horizons = np.arange(0, 49)  # 0 to 48 hours
    testset = testset.copy()
    testset["timestamp"] = pd.to_datetime(testset["timestamp"])
    testset["pv_id"] = testset["pv_id"].astype(str)

    # Cross-join testset with horizons — one row per (test_entry, horizon)
    expanded = testset.loc[testset.index.repeat(len(horizons))].reset_index(drop=True)
    expanded["horizon_hour"] = np.tile(horizons, len(testset))
    expanded["timestamp"] = expanded["timestamp"] + pd.to_timedelta(expanded["horizon_hour"], unit="h")

    # Batch-extract values from xarray: group by pv_id to minimize dataset access
    print(f"Extracting {len(expanded)} values across {expanded['pv_id'].nunique()} PV systems")
    values = np.full(len(expanded), np.nan)

    for pv_id, group in expanded.groupby("pv_id"):
        if pv_id not in pv_ds:
            continue

        pv_series = pv_ds[pv_id]
        available_times = pd.DatetimeIndex(pv_series.datetime.values)

        # Find which requested timestamps exist in the dataset
        requested_times = group["timestamp"].values
        mask = np.isin(requested_times, available_times)

        if mask.any():
            matched_times = requested_times[mask]
            selected = pv_series.sel(datetime=matched_times).values / 1000  # W to kW
            values[group.index[mask]] = selected

    expanded["value"] = values

    print(f"Extracted {np.isfinite(values).sum()} valid values out of {len(values)} total")
    return expanded[["pv_id", "timestamp", "value", "horizon_hour"]]
