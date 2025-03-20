"""Data loading, cleaning, augmenting, etc. of the `uk_pv` dataset."""

import logging

import pandas as pd

_log = logging.getLogger(__name__)


class C:
    lat = "latitude"
    lon = "longitude"
    date = "timestamp"
    power = "generation_wh"
    id = "ss_id"
    orientation = "orientation"
    tilt = "tilt"
    factor = "factor"


def filter_rows(pv: pd.DataFrame, mask: pd.Series, text: str | None = None):
    """Filter a dataframe and print how much was removed."""
    n1 = len(pv)
    pv = pv[mask]
    n2 = len(pv)

    s = f"Removed {n1 - n2} ({(n1 - n2) / n1 * 100:.1f}%) rows."
    if text:
        s += f" [{text}]"
    _log.info(s)

    return pv


def trim_pv(pv: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # Remove all the zero "power" values: there are quite a lot of those.
    pv = filter_rows(pv, pv[C.power] > 0.1, "power > 0.1")

    ss_ids = meta[C.id].unique()

    pv = filter_rows(pv, pv[C.id].isin(ss_ids), "unknown ss_id")

    return pv


def get_max_power_for_time_of_day(
    df: pd.DataFrame, *, radius: int = 7, min_records: int = 0
) -> pd.DataFrame:
    """For each data point, find the max in a timewindow, at the same time of day.

    Arguments:
    ---------
        df: index: [ss_id, timestamp], columns: [power]
        radius: How many days before and after to look at.

    Return:
    ------
        A dataframe with the same index (but sorted!) and the max power, keeping the same column
        name.

    See the test case for an example.
    """
    df = df.reset_index(1).copy()
    df["time"] = df[C.date].dt.time
    df = df.set_index(["time", C.date], append=True, drop=False)
    # Now index is: ss_id, time, datetime

    df = df.sort_index()

    # This is where the magic happens: group by ss_id and time_of_day, then do a rolling max on the
    # days.
    df = (
        df.groupby(
            [pd.Grouper(level=0), pd.Grouper(level=1)],
        )
        .rolling(
            f"{1 + radius * 2}D",
            on=C.date,
            center=True,
            min_periods=min_records,
            closed="both",
        )
        .max()
    )

    # Reshape and sort by index.
    df = df.reset_index(level=(1, 2, 3), drop=True).sort_index()

    return df
