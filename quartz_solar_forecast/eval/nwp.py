""" Get nwp data from HF"""
import os
import pandas as pd

import ocf_blosc2  # noqa
import xarray as xr
from huggingface_hub import HfFileSystem


def get_nwp(time_locations: pd.DataFrame):
    """
    Get all the nwp data fpr the time locations

    time_locations should have the following columns:
    - timestamp
    - latitude
    - longitude
    - pv_id
    """

    all_nwp_dfs = []
    for i, row in time_locations.iterrows():
        print(f"{i} of {len(time_locations)}")
        one_nwp_df = get_nwp_for_one_timestamp_one_location(
            row["timestamp"], row["latitude"], row["longitude"]
        )

        one_nwp_df["timestamp"] = row["timestamp"]
        one_nwp_df["pv_id"] = row["pv_id"]
        one_nwp_df["latitude"] = row["latitude"]
        one_nwp_df["longitude"] = row["longitude"]

        all_nwp_dfs.append(one_nwp_df)

    all_nwp_df = pd.concat(all_nwp_dfs)

    return all_nwp_df


def get_nwp_for_one_timestamp_one_location(timestamp: pd.Timestamp, latitude, longitude):
    """
    Get NWP data from Hugging Face for one timestamp and one location

    :param timestamp: the timestamp for when you want the forecast for
    :param latitude: the latitude of the location
    :param longitude: the longitude of the location

    :return: nwp forecast in xarray
    """

    # TODO add caching

    fs = HfFileSystem()
    # List which files are available. Not all dates, and model run times are available
    # print(fs.ls("datasets/openclimatefix/dwd-icon-eu/data/2022/4/11/", detail=False))

    # round timestamp to 6 hours floor
    timestamp = timestamp.floor("6H")
    year = timestamp.year
    month = timestamp.month
    day = timestamp.day
    date_and_hour = timestamp.strftime("%Y%m%d_%H")

    date = f"{year}/{month}/{day}"
    file_location = f"{date}/{date_and_hour}"
    huggingface_route = "zip:///::hf://datasets/openclimatefix/dwd-icon-eu/data"
    # huggingface_route = "datasets/openclimatefix/dwd-icon-eu/data"
    huggingface_file = f"{huggingface_route}/{file_location}.zarr.zip"

    # dataset variables
    variables = ["t_2m", "tot_prec", "clch", "clcm", "clcl", "u", "v", "aswdir_s", "aswdifd_s"]

    cache_dir = "data/nwp"
    cache_file = f"{cache_dir}/{file_location}_{latitude}_{longitude}.zarr"
    if not os.path.exists(cache_file):
        # use fsspec to copy file
        print(f"Opening file {huggingface_file} from HF to local")

        data = xr.open_zarr(
            f"{huggingface_file}",
            chunks="auto",
        )

        # take nearest location and only select the variables we want
        data_at_location = data.sel(latitude=latitude, longitude=longitude, method="nearest")
        data_at_location = data_at_location[variables]

        # choise the first isobaricInhPa
        data_at_location = data_at_location.isel(isobaricInhPa=-1)

        #  reduce to 54 hours timestaps, this means there is at least a 48 hours forecast
        data_at_location = data_at_location.isel(step=slice(0, 54))

        # load all the data, this can take about ~1 minute seconds
        print(f"Loading dataset for {timestamp=} {longitude=} {latitude=}")
        data_at_location.load()

        # save to cache
        data_at_location.to_zarr(cache_file)
    else:
        # load from cache
        print("loading from cache")
        data_at_location = xr.open_zarr(cache_file)

    # make times from the init time + steps
    times = pd.to_datetime(data_at_location.time.values) + pd.to_timedelta(
        data_at_location.step.values, unit="h"
    )

    # convert to pandas dataframe
    df = pd.DataFrame(times, columns=["timestamp"])
    for variable in variables:
        print(variable)
        df[variable] = data_at_location[variable].values

    # make wind speed out of u and v
    df["windspeed_10m"] = (df["u"] ** 2 + df["v"] ** 2) ** 0.5

    # rename variables
    df = df.rename(
        columns={
            "t_2m": "temperature_2m",
            "tot_prec": "precipitation",
            "aswdifd_s": "shortwave_radiation",
            "aswdir_s": "direct_radiation",
            "clcl": "cloudcover_low",
            "clcm": "cloudcover_mid",
            "clch": "cloudcover_high",
        }
    )

    # add visbility for the moment
    # TODO
    df["visibility"] = 10000

    # drop u and v
    df = df.drop(columns=["u", "v"])

    return df
