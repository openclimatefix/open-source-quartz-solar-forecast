"""Infer the orientation and tilt of pv panels from their data.

NOTE: We are assuming that the data is in Wh/5min.
NOTE: This is still somewhat specific to the uk_pv dataset but it could be generalized.
"""
# TODO Fix the assumption above: we should probably treat everything in kW.

import argparse
import pathlib

import numpy as np
import pandas as pd
import scipy
import tqdm

from psp.clients.uk_pv.data import C, get_max_power_for_time_of_day
from psp.pv import get_irradiance


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-d", "--data", type=pathlib.Path, help="parquet data file", required=True)
    parser.add_argument("-m", "--meta", type=pathlib.Path, help="metadata file", required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="output file with the infered metadata",
        required=True,
    )
    return parser.parse_args()


def _infer_params(
    df: pd.DataFrame,
    ss_id: int,
    lat: float,
    lon: float,
    learn_normalisation: bool = True,
) -> dict | None:
    """Find the best maching parameters (tilt, orientation, capacity) for a given PV system."""
    try:
        df = df.loc[(ss_id, slice(None)), :]
    except KeyError:
        return None

    if len(df) == 0:
        return None

    # Get the capacity.
    capacity = df[C.power].quantile(0.99)

    max_power = get_max_power_for_time_of_day(df[[C.power]], radius=9, min_records=10)

    # We do a little bit of something on our max power data.
    smooth = (
        max_power.reset_index()
        .groupby(C.id)
        .rolling("1h", on=C.date, center=True, min_periods=4, closed="both")
        .mean()
        .reset_index()
        .set_index([C.id, C.date], drop=False)
        .drop(columns="level_1")
        .sort_index()
    )

    smooth = smooth[~smooth[C.power].isnull()]

    # We'll only consider few dates.
    smooth["date"] = smooth[C.date].dt.date
    # We want those dates to have some data.
    date_counts = smooth.groupby("date")[[C.power]].count()
    date_counts = date_counts[date_counts[C.power] > 50]
    # Select N dates evenly.
    num_dates = 20
    if len(date_counts) < 20:
        return None
    indices = np.round(np.linspace(0, len(date_counts) - 1, num_dates)).astype(int)
    date_counts = date_counts.iloc[indices]
    dates = date_counts.index

    # Keep only the data for those dates.
    data = smooth[smooth["date"].isin(dates)]
    timestamps = data.index.get_level_values(1)

    # Only consider the "power" column.
    data = data[C.power]

    # Normalize the data.
    if not learn_normalisation:
        data = data / data.max()

    # Now define our objective function that we want to minimze.
    def cost(params, lat: float, lon: float, timestamps: pd.DatetimeIndex):
        tilt, orientation, factor = params
        irr = get_irradiance(
            lat=lat,
            lon=lon,
            timestamps=timestamps,
            tilt=tilt,
            orientation=orientation,
        )
        ref = irr["poa_global"]

        if not learn_normalisation:
            ref = ref / ref.max()

        return ((data - ref * factor) ** 2).mean()

    result = scipy.optimize.minimize(
        cost,
        # TODO 0.1 is a heuristic that probably needs revisiting.
        [30, 180, capacity * 0.1 if learn_normalisation else 1],
        bounds=[(0, 90), (0, 360), (0, None) if learn_normalisation else (1, 1)],
        args=(lat, lon, timestamps),
    )

    tilt, orientation, factor = result.x
    fit_err = result.fun

    return dict(
        tilt=tilt,
        orientation=orientation,
        factor=factor,
        capacity=capacity,
        fit_err=fit_err,
    )


def main():
    args = parse_args()

    meta = pd.read_csv(args.meta)
    meta = meta.set_index(C.id)

    df = pd.read_parquet(args.data)[[C.power]]

    # Convert the units from Wh/5min to kW.
    # TODO We should find a way to have kW as input.
    df = df * 12 / 1000

    ids = df.index.get_level_values(0).unique().tolist()
    df[C.id] = df.index.get_level_values(0)
    df[C.date] = df.index.get_level_values(1)

    new_data = []

    for ss_id in tqdm.tqdm(ids):
        meta_row = meta.loc[ss_id]
        lat = meta_row[C.lat]
        lon = meta_row[C.lon]
        new_meta = _infer_params(df, ss_id=ss_id, lat=lat, lon=lon, learn_normalisation=True)
        if new_meta is not None:
            new_data.append({"ss_id": ss_id, **new_meta})

    out = pd.DataFrame.from_records(new_data)
    out.to_csv(
        args.output,
        float_format=lambda x: f"{x:.2f}",
        index=False,
    )


if __name__ == "__main__":
    main()
