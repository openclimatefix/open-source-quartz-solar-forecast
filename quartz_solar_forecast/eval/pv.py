import pyarrow.dataset as ds

import os

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem
import glob

fs = HfFileSystem()


def get_pv_metadata(testset: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata (lat/lon/capacity) with testset of pv_id + timestamp."""
    cache_dir = "data/pv"
    metadata_file = f"{cache_dir}/metadata.csv"

    if not os.path.exists(metadata_file) or os.path.getsize(metadata_file) == 0:
        os.makedirs(cache_dir, exist_ok=True)
        fs.get("datasets/openclimatefix/uk_pv/metadata.csv", metadata_file)

    metadata_df = pd.read_csv(metadata_file)

    # align schema
    metadata_df = metadata_df.rename(columns={"ss_id": "pv_id"})

    combined_data = testset.merge(metadata_df, on="pv_id", how="left")

    # keep only useful columns
    combined_data = combined_data[
        ["pv_id", "timestamp", "latitude_rounded", "longitude_rounded", "kWp"]
    ].rename(
        columns={
            "latitude_rounded": "latitude",
            "longitude_rounded": "longitude",
            "kWp": "capacity",
        }
    )

    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
    return combined_data


def get_pv_truth(testset: pd.DataFrame, horizon_hours: int = 48) -> pd.DataFrame:
    """Fetch PV generation truth values for given testset.
    Vectorized across pv_id and horizons.
    """
    print("Loading PV data")

    cache_dir = "data/pv"
    parquet_dir = f"{cache_dir}/30_minutely"

    if not os.path.exists(parquet_dir):
        print("Downloading PV parquet data from HF...")
        os.makedirs(cache_dir, exist_ok=True)
        fs.get("datasets/openclimatefix/uk_pv/30_minutely", cache_dir, recursive=True)

    # Find all non-empty parquet files
    files = glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)
    non_empty_files = [f for f in files if os.path.getsize(f) > 0]

    if not non_empty_files:
        raise FileNotFoundError("No valid parquet files found (all are empty).")

    # Load dataset only from non-empty files
    dataset = ds.dataset(non_empty_files, format="parquet")
    pv_data = dataset.to_table().to_pandas()

    # Ensure datetime column is parsed
    pv_data["datetime_GMT"] = pd.to_datetime(pv_data["datetime_GMT"]).dt.tz_convert("UTC")

    # Vectorized expansion
    horizons = np.arange(horizon_hours + 1)
    expanded = testset.loc[testset.index.repeat(len(horizons))].copy()
    expanded["horizon_hour"] = np.tile(horizons, len(testset))

    expanded["timestamp"] = pd.to_datetime(expanded["timestamp"]).dt.tz_localize("UTC")
    # Merge with pv_data
    merged = expanded.merge(
        pv_data,
        left_on=["pv_id", "timestamp"],
        right_on=["ss_id", "datetime_GMT"],
        how="left",
    )
    # Convert to kWh
    merged["value"] = merged["generation_Wh"] / 1000.0

    result = merged[["pv_id", "timestamp", "value", "horizon_hour"]].copy()
    return result
