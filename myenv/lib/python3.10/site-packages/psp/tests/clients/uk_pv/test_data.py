import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from psp.clients.uk_pv.data import C, get_max_power_for_time_of_day


def _ts(d, h):
    return pd.Timestamp(year=2022, month=1, day=d, hour=h)


def _from_records(rec):
    # Make sure the ids are strings.
    rec = [[str(x[0])] + x[1:] for x in rec]

    return pd.DataFrame.from_records(rec, columns=[C.id, C.date, C.power]).set_index([C.id, C.date])


def test_get_max_power_for_time_of_day():
    df = _from_records(
        [
            [1, _ts(1, 1), 1.0],
            [1, _ts(1, 2), 3],
            [1, _ts(2, 1), 5],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 9],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 4],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ],
    )

    max_0 = get_max_power_for_time_of_day(df, radius=0)

    assert_frame_equal(max_0, df)

    max_1 = get_max_power_for_time_of_day(df, radius=1)

    expected = _from_records(
        [
            [1, _ts(1, 1), 5.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 9],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ]
    )

    assert_frame_equal(max_1, expected)

    max_2 = get_max_power_for_time_of_day(df, radius=2)

    expected = _from_records(
        [
            [1, _ts(1, 1), 9.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 10],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ],
    )

    assert_frame_equal(max_2, expected)

    # Test the `min_records` parameter.
    max_2_records = get_max_power_for_time_of_day(df, radius=2, min_records=2)

    expected = _from_records(
        [
            [1, _ts(1, 1), 9.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 10],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), np.nan],
            [2, _ts(2, 1), 30],
        ],
    )

    assert_frame_equal(max_2_records, expected)
