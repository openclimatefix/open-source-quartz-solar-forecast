import os

import pandas as pd
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
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])

    return combined_data
