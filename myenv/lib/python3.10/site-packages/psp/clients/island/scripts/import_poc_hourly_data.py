"""Load the raw hourly "island" dataset from the POC."""

import argparse
import pathlib

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="folder with excel files",
    )
    parser.add_argument("output", type=pathlib.Path, help="output netcdf file (.nc)")

    return parser.parse_args()


def load_all_hourly(directory: pathlib.Path) -> pd.DataFrame:
    """Load the data from all .xlsx files in `directory`."""
    paths = directory.glob("*.xlsx")

    # Read all Excel files into a list of dataframes.
    dataframes = []

    for path in paths:
        df = pd.read_excel(path, engine="openpyxl")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values("Date")

    return df


def transpose_data(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the column hours into rows.

    In the original the days are on the rows and the hours on the columns.
    In this function we move the hours to the rows.
    """
    # The hour columns are the digit ones.
    hour_columns = [col for col in df.columns if str(col).isdigit()]

    df = df.melt(
        id_vars=[
            "Date",
            " Total Max Capacity of Read Meters/KW",
            "Total Max Capacity",
            "Number of Read Meters",
            "Total Number of Meters",
        ],
        value_vars=hour_columns,
        var_name="Hour",
        value_name="power",
    )
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = pd.to_timedelta(df["Hour"], unit="h")

    # Make one timestamp with days and hours.
    df["ts"] = df["Date"] + df["Hour"]

    df = df.sort_values("ts")

    return df


def main():
    args = _parse_args()

    df = load_all_hourly(args.input)

    df = transpose_data(df)

    df = df.rename(
        columns={
            "Total Max Capacity": "capacity",
        }
    )

    # From kW to MW
    df["capacity"] = df["capacity"] / 1000.0
    df["power"] = df["power"] / 1000.0

    # Only keep the columns that we really need.
    df = df[["ts", "power", "capacity"]]

    # Convert timezones to UTC.
    df["ts"] = (
        df["ts"]
        .dt.tz_localize("Europe/Malta", nonexistent="NaT", ambiguous="NaT")
        .dt.tz_convert(None)
    )

    # Some datetimes might be NaT after the timezone conversion.
    df = df[~pd.isna(df["ts"])]

    df = df.sort_values("ts")

    df = df.set_index("ts")
    ds = df.to_xarray()

    # We only have one time series but we still give it an id.
    ds = ds.expand_dims(pv_id=["0"])

    # Add hard-coded lat/lon coordinates.
    ds = ds.assign_coords(latitude=("pv_id", [35.87420752836937]))
    ds = ds.assign_coords(longitude=("pv_id", [14.451608933898406]))

    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
