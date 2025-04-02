import datetime as dt

import numpy as np
import pytest
import xarray as xr

from psp.models.base import PvSiteModelConfig
from psp.models.historical_forecasts import HistoricalForecasts
from psp.typings import Horizons, X, Y

D0 = dt.datetime(2020, 1, 1, 0)
H = dt.timedelta(hours=1)


@pytest.mark.parametrize(
    "pv_id,ts,expected",
    [
        ["a", D0, [np.nan, np.nan, 0, 1]],
        # Same forecast but we are one hour later so the horizons are dephased.
        ["a", D0 + H, [np.nan, 0, 1, np.nan]],
        # 2 hours later we hit the next forecast
        ["a", D0 + 2 * H, [np.nan, np.nan, 2, 3]],
        ["a", D0 + 3 * H, [np.nan, 2, 3, np.nan]],
        ["a", D0 + 4 * H, [2, 3, np.nan, np.nan]],
        ["a", D0 + 5 * H, [3, np.nan, np.nan, np.nan]],
        ["a", D0 + 6 * H, [np.nan, np.nan, np.nan, np.nan]],
        #
        ["b", D0, [np.nan, np.nan, 4, 5]],
        ["b", D0 + H, [np.nan, 4, 5, np.nan]],
        ["b", D0 + 2 * H, [np.nan, np.nan, 6, 7]],
        ["b", D0 + 3 * H, [np.nan, 6, 7, np.nan]],
    ],
)
def test_historical_forecast_model(pv_id, ts, expected):
    da = xr.DataArray(
        data=np.arange(8).reshape(2, 2, 2),
        dims=("pv_id", "time", "step"),
        coords={
            "pv_id": ("pv_id", ["a", "b"]),
            "time": ("time", [D0, D0 + 2 * H]),
            "step": ("step", [H * 2, H * 3]),
        },
    )

    ds = xr.Dataset({"power": da})

    model = HistoricalForecasts(
        config=PvSiteModelConfig(horizons=Horizons(duration=60, num_horizons=4)),
        data_or_path=ds,
    )

    assert model.predict(X(pv_id=pv_id, ts=ts)) == Y(powers=np.array(expected, dtype=float))
