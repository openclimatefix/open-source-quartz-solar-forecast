import datetime

import numpy as np
import pandas as pd


def to_pydatetime(x: np.datetime64) -> datetime.datetime:
    """Convert a numpy datetime to a python datetime."""
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    return pd.Timestamp(x).to_pydatetime()
