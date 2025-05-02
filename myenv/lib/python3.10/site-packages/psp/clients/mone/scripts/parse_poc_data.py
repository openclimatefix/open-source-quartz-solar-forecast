"""Parse the initial Mone raw dataset."""

import argparse
import pathlib

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

    # Load data from the csv files.
    csv_files = [x for x in pathlib.Path(args.input).iterdir() if x.suffix == ".csv"]
    dfs = [pd.read_csv(x, parse_dates=["timestamp"]) for x in csv_files]
    dfs = [df.set_index("timestamp") for df in dfs]

    # There were some duplicate timestamps: let's get rid of them.
    for i, df in enumerate(dfs):
        dup_index = df.index.duplicated()
        dfs[i] = df[~dup_index]

    # Merge all the individual dataframes.
    df = pd.concat(dfs, axis=1)

    # Use the individual file names as pv_ids.
    names = [x.stem for x in csv_files]
    df.columns = pd.Index(names)

    # Pivot the data.
    s = df.stack()
    s.index.names = ["ts", "pv_id"]

    # Make it into a data array.
    da = s.to_xarray()

    # Add the lat/lon that we googled for each location in the provided Location.txt.
    city = pd.DataFrame(index=names, columns=["lat", "lon"])

    city.loc["16008274"][["lat", "lon"]] = (51.2869, -0.7526)
    city.loc["12058624"][["lat", "lon"]] = (51.5072, 0.1276)
    city.loc["12078299"][["lat", "lon"]] = (52.6383, 1.5506)
    city.loc["12084263"][["lat", "lon"]] = (52.3024, 0.6940)
    city.loc["12037687"][["lat", "lon"]] = (55.7832, -3.9811)
    city.loc["16031157"][["lat", "lon"]] = (53.8008, -1.5491)
    # This one we are less sure about.
    city.loc["16000872"][["lat", "lon"]] = (51.45000076, -1.72939467)
    city.loc["16042597"][["lat", "lon"]] = (50.8376, -0.7749)

    # Make lists of lat and lon that matches the order of pv_id in the data array.
    lat = [city.loc[pv_id, "lat"] for pv_id in list(da.coords["pv_id"].values)]
    lon = [city.loc[pv_id, "lon"] for pv_id in list(da.coords["pv_id"].values)]

    # Add the coordinates.
    da = da.assign_coords(latitude=("pv_id", lat))
    da = da.assign_coords(longitude=("pv_id", lon))

    # Wrap the DataArray into a Dataset.
    ds = xr.Dataset(dict(power=da))

    # Save!
    ds.to_netcdf(args.output)

    # If you need a 5 minutely version of the data, here is how to do it:
    # ds5 = ds.resample(ts='5min', loffset=dt.timedelta(seconds=60 * 2.5)).mean()


if __name__ == "__main__":
    main()
