import datetime as dt

import pytest

from psp.dataset import DateSplits, TrainDateSplit, auto_date_split
from psp.dataset import TestDateSplit as _TestDateSplit

# Default split dates for our PV data source fixture.
D0 = dt.datetime(2020, 1, 1)
D1 = dt.datetime(2020, 1, 31)

D0_minus1 = D0 - dt.timedelta(days=1)


@pytest.mark.parametrize(
    "num_trainings,expected_train_dates",
    [
        [1, [dt.datetime(2019, 12, 31)]],
        [2, [dt.datetime(2019, 12, 31), dt.datetime(2020, 1, 15)]],
    ],
)
def test_auto_date_split(num_trainings, expected_train_dates):
    train_days = 10
    splits = auto_date_split(D0, D1, num_trainings=num_trainings, train_days=train_days)

    assert splits == DateSplits(
        train_date_splits=[
            TrainDateSplit(train_date=d, train_days=train_days) for d in expected_train_dates
        ],
        test_date_split=_TestDateSplit(start_date=D0, end_date=D1),
    )
