import pickle
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from psp.data_sources.pv import NetcdfPvDataSource


def _make_pv_data_xarray() -> xr.Dataset:
    pv_ids = [1, 2, 3]
    t0 = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    ts_range = [t0 + i * dt for i in range(4)]

    n = len(pv_ids)
    m = len(ts_range)

    # (pv_id, timestamp)
    data = np.arange(n * m, dtype=float).reshape(n, m)

    da = xr.DataArray(
        data=data,
        coords={
            # Using the same column names as in the original dataset.
            "pv_id": pv_ids,
            "ts": ts_range,
        },
        dims=["pv_id", "ts"],
    )
    d = xr.Dataset({"power": da})
    return d


@pytest.fixture
def pv_data_source(tmp_path):
    d = _make_pv_data_xarray()
    path = tmp_path / "pv_data_source.netcdf"
    d.to_netcdf(path)
    return NetcdfPvDataSource(
        path,
    )


def test_pv_data_source_rename(tmp_path):
    d = _make_pv_data_xarray()
    d = d.rename({"pv_id": "mon_id", "ts": "mon_ts", "power": "mon_power"})
    ds = NetcdfPvDataSource(
        d,
        id_dim_name="mon_id",
        timestamp_dim_name="mon_ts",
        rename={"mon_power": "power"},
    )

    assert (
        ds.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))["power"].size
        == 2
    )


@pytest.mark.parametrize(
    "at,lag,expected_max_ts,expected_size",
    [
        [None, 0, datetime(2023, 1, 4), 2],
        [None, 10, datetime(2023, 1, 4), 2],
        [None, 24 * 60, datetime(2023, 1, 4), 2],
        [datetime(2023, 1, 3), 0, datetime(2023, 1, 2, 23, 59, 59), 1],
        [datetime(2023, 1, 3), 10, datetime(2023, 1, 2, 23, 49, 59), 1],
        [datetime(2023, 1, 3), 24 * 60, datetime(2023, 1, 1, 23, 59, 59), 0],
    ],
)
def test_pv_data_source_as_available_at(at, lag, expected_max_ts, expected_size):
    data = _make_pv_data_xarray()
    pv_data_source = NetcdfPvDataSource(data, lag_minutes=lag)

    # Before `as_available_at`.
    assert pv_data_source.min_ts() == datetime(2023, 1, 1)
    assert pv_data_source.max_ts() == datetime(2023, 1, 4)
    assert pv_data_source.list_pv_ids() == "1 2 3".split()
    assert (
        pv_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == 2
    )

    # With `as_available_at`.
    if at is not None:
        new_data_source = pv_data_source.as_available_at(at)
    else:
        new_data_source = pv_data_source
    assert new_data_source.min_ts() == datetime(2023, 1, 1)
    assert new_data_source.max_ts() == expected_max_ts
    assert (
        new_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == expected_size
    )

    # The old data source still works as usual.
    assert (
        pv_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == 2
    )


def test_ignore_pv_ids(tmp_path):
    d = _make_pv_data_xarray()
    ds = NetcdfPvDataSource(d, ignore_pv_ids=["2", "3"])

    assert ds.list_pv_ids() == ["1"]

    with pytest.raises(KeyError):
        ds.get(pv_ids=["2"])

    with pytest.raises(KeyError):
        ds.get(pv_ids=["1", "3"])


def test_non_path_pv_data_source_pickle_raises(tmp_path):
    d = _make_pv_data_xarray()
    ds = NetcdfPvDataSource(d)
    path = tmp_path / "test_non_path_pv_data_source_pickle_raises.pkl"
    with open(path, "wb") as f:
        with pytest.raises(RuntimeError) as e:
            pickle.dump(ds, f)
        assert "that were constructed using a path" in str(e)
