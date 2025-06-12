"""Import historical forecast from the "island" client and save them as a single netcdf file.

The resulting file can be loadeding into a `HistoricalForecasts` model.
"""

import argparse
import datetime as dt
import pathlib
from warnings import warn

import pandas as pd
import tqdm
from openpyxl import load_workbook


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="Directory with excel files.",
    )
    parser.add_argument("output", type=pathlib.Path, help="Output netcdf file (.nc).")
    parser.add_argument(
        "--max", type=int, help="Max number of files to process. Useful for debugging."
    )

    return parser.parse_args()


def _parse_excel(file_path: pathlib.Path, check_date=dt.datetime) -> pd.DataFrame:
    """Load data from an excel file.

    check_date: The date we expect to find in the file.
    """
    wb = load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active

    # Cell A1 contains the date.
    day = pd.to_datetime(ws["A1"].value, format="%Y%m%d")

    # This particular file has the wrong date in it.
    if check_date == dt.datetime(2023, 2, 20):
        day = dt.datetime(2023, 2, 20)

    # Sanity check.
    if day != check_date:
        warn(f'data date "{day}" does not match file name date "{check_date}"')

    # Extract the data from the specified cells.
    powers = []
    dates = []
    for i, row in enumerate(range(6, 30)):
        # -1 because they are the end of the time window.
        hour = ws["A" + str(row)].value - 1
        if hour != i:
            warn(f"row {i} != hour {hour} for date {check_date}")
        # Use `i` because it is more consistent.
        date = pd.to_datetime(day + pd.DateOffset(hours=i))

        power = ws["E" + str(row)].value

        dates.append(date)
        powers.append(power)

    df = pd.DataFrame({"power": powers, "ts": dates})

    # Assume the forecast was made yesterday at 10am (this is what the client told us).
    df["time"] = (day - pd.DateOffset(days=1)).replace(hour=10)
    df["step"] = df["ts"] - df["time"]
    del df["ts"]

    # Make everything UTC, assuming the timezone is malta.
    df["time"] = (
        df["time"]
        .dt.tz_localize("Europe/Malta", nonexistent="NaT", ambiguous="NaT")
        .dt.tz_convert(None)
    )

    # Remove the potential NaT.
    df = df[~pd.isna(df["time"])]

    return df


def _get_datetime_from_path(path: pathlib.Path) -> dt.datetime:
    """Extract the date from the name of the file."""
    name = path.stem
    date = name.split(" ")[-1]
    day, month, year = date.split("_")
    return dt.datetime(int(year), int(month), int(day))


def main():
    args = _parse_args()

    paths = list(args.input.glob("*.xlsx"))

    paths = sorted(paths, key=_get_datetime_from_path)

    if args.max is not None:
        paths = paths[: args.max]

    dfs = []

    seen_dates = set()

    for path in tqdm.tqdm(paths):
        date = _get_datetime_from_path(path)
        if date in seen_dates:
            raise RuntimeError(f"date already seen: {date}")
        seen_dates.add(date)
        df = _parse_excel(path, check_date=date)
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sort_values(["time", "step"])
    df = df.set_index(["time", "step"])

    da = df.to_xarray()

    # Add a trivial pv id, the same as when we import the ground truth PV data.
    da = da.expand_dims(pv_id=["0"])

    da.to_netcdf(args.output)


if __name__ == "__main__":
    main()
