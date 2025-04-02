import argparse
import pathlib

import numpy as np
import pandas as pd
import xarray as xr


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input", help="directory containing the original .csv files from the client"
    )
    parser.add_argument("output", help="output netcdf file")
    return parser.parse_args()


def main():
    args = _parse_args()

    csv_files = [x for x in pathlib.Path(args.input).iterdir() if x.suffix == ".csv"]

    dfs = [pd.read_csv(x, parse_dates=["UtcDatetime"]) for x in csv_files]

    df = pd.concat(dfs, ignore_index=True)

    ds = xr.Dataset.from_dataframe(df)
    ds = ds.set_index(index=["UtcDatetime", "SiteId"]).unstack("index")

    names = np.unique(ds.coords["SiteId"].values)
    names = [str(name) for name in names]
    site_ids = [str(site_id) for site_id in ds.coords["SiteId"].values]

    # Add the lat/lon that we googled for each location in the provided Location.txt.
    city = pd.DataFrame(index=names, columns=["lat", "lon"])
    city.loc["4353"][["lat", "lon"]] = (51.17533, 0.40693)
    city.loc["4403"][["lat", "lon"]] = (51.65856, -1.79106)
    city.loc["4416"][["lat", "lon"]] = (50.79061, -1.85936)
    city.loc["4417"][["lat", "lon"]] = (50.79061, -1.85936)
    city.loc["4423"][["lat", "lon"]] = (51.64675, -1.80934)
    city.loc["6331"][["lat", "lon"]] = (51.97173, 0.36421)
    city.loc["6332"][["lat", "lon"]] = (51.82652, 0.72746)
    city.loc["6378"][["lat", "lon"]] = (52.61083, -1.56508)
    city.loc["6381"][["lat", "lon"]] = (52.21649, -1.4398)
    city.loc["6385"][["lat", "lon"]] = (53.02505, -0.73305)
    city.loc["6590"][["lat", "lon"]] = (51.44815, -1.98928)
    city.loc["6592"][["lat", "lon"]] = (50.85989, -0.7939)
    city.loc["6594"][["lat", "lon"]] = (51.75541, -1.64461)
    city.loc["6596"][["lat", "lon"]] = (51.63601, -1.39277)
    city.loc["6600"][["lat", "lon"]] = (51.48668, -1.559777)
    city.loc["6605"][["lat", "lon"]] = (51.38238, -2.0898)
    city.loc["6670"][["lat", "lon"]] = (50.82233, -4.42377)
    city.loc["25842"][["lat", "lon"]] = (51.75541, -1.64461)
    city.loc["27855"][["lat", "lon"]] = (51.65529, -1.16366)

    # Make lists of lat and lon that matches the order of pv_id in the data array.
    lat = [city.loc[SiteId, "lat"] for SiteId in site_ids]
    lon = [city.loc[SiteId, "lon"] for SiteId in site_ids]

    # Add the coordinates.
    ds = ds.assign_coords(latitude=("SiteId", lat))
    ds = ds.assign_coords(longitude=("SiteId", lon))

    ds = ds.drop_vars(["Unnamed: 0", "PostCode"])
    ds = ds.rename({"UtcDatetime": "ts", "SiteId": "pv_id"})
    ds = ds.rename({"Actual_MW": "power", "CapacityMw": "capacity"})

    # Drop any NaNs
    # ds = ds.dropna(dim="ts", subset=["power"])

    ds = ds.dropna("ts", how="all")

    # Save!
    ds.to_netcdf(args.output)

    # If you need a 5 minutely version of the data, here is how to do it:
    # ds5 = ds.resample(ts='5min', loffset=dt.timedelta(seconds=60 * 2.5)).mean()


if __name__ == "__main__":
    main()
