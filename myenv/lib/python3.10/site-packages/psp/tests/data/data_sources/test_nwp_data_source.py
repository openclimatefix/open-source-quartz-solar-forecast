from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from psp.data_sources.nwp import NwpDataSource
from psp.gis import CoordinateTransformer
from psp.utils.dates import to_pydatetime

T0 = datetime(2023, 1, 1, 0)
T1 = datetime(2023, 1, 1, 1)
T2 = datetime(2023, 1, 1, 2)

LON0 = -2
LON1 = -1
LON2 = 0

LAT0 = 50
LAT1 = 51
LAT2 = 52


@pytest.fixture(params=[27700, 4326])
def nwp_data_sources(tmp_path, request):
    coord_system: int = request.param

    lats = [LAT0, LAT1, LAT2]
    lons = [LON0, LON1, LON2]

    transform_coords = CoordinateTransformer(4326, coord_system)

    new_coords = transform_coords(zip(lats, lons))

    xs = [p[0] for p in new_coords]
    ys = [p[1] for p in new_coords]

    times = [T0, T1, T2]

    # Predictions for the next 2 hours (including one for right now).
    steps = [timedelta(0), timedelta(hours=1), timedelta(hours=2)]

    variables = "a b c".split()

    data = np.arange(len(xs) * len(ys) * len(times) * len(steps) * len(variables)).reshape(
        len(xs),
        len(ys),
        len(times),
        len(steps),
        len(variables),
    )

    coords = {
        "x": list(xs),
        "y": list(ys),
        "time": times,
        "step": steps,
        "variable": variables,
    }

    da = xr.DataArray(data=data, coords=coords)

    ds = xr.Dataset({"value": da})

    path = tmp_path / "nwp_fixture.zarr"
    ds.to_zarr(path)

    return NwpDataSource(path, coord_system=coord_system)


def hours(x: float) -> timedelta:
    return timedelta(hours=x)


@pytest.mark.parametrize(
    "now,ts,expected_time,expected_step",
    [
        [T0, T0, T0, 0],
        [T0, T0 + hours(1), T0, 1],
        [T0, T0 + hours(0.51), T0, 1],
        [T0, T0 + hours(0.49), T0, 0],
        [T0, T0 + hours(1.49), T0, 1],
        [T0, T0 + hours(10), T0, 2],
        # "now" is almost T1 but not quite: it means we still can't access it because it's in the
        # future.
        [T0 + hours(0.9), T0 + hours(1), T0, 1],
        [T0 + hours(0.9), T0 + hours(2), T0, 2],
        [T0 + hours(0.9), T0 + hours(1.49), T0, 1],
        [T0 + hours(0.9), T0 + hours(1.51), T0, 2],
    ],
)
def test_nwp_data_source_check_times_one_step(
    now, ts, expected_time, expected_step, nwp_data_sources
):
    data = nwp_data_sources.get(now=now, timestamps=ts)

    # Always one init_time, one step, 3 variables, and 3x3 lat/lon.
    assert data.size == 3 * 3 * 3
    assert to_pydatetime(data.coords["time"].values) == expected_time
    assert data.coords["step"].values == np.timedelta64(expected_step, "h")


@pytest.mark.parametrize(
    "now,ts,expected_time,expected_steps",
    [
        [T0, [T0], T0, [0]],
        [T0, [T0 + hours(1)], T0, [1]],
        [T0, [T0 + hours(0.51)], T0, [1]],
        [T0, [T0 + hours(0.49)], T0, [0]],
        [T0, [T0 + hours(1.49)], T0, [1]],
        [T0, [T0 + hours(10)], T0, [2]],
        # # "now" is almost T1 but not quite: it means we still can't access it because it's in the
        # # future.
        [T0 + hours(0.9), [T0 + hours(1)], T0, [1]],
        [T0 + hours(0.9), [T0 + hours(2)], T0, [2]],
        [T0 + hours(0.9), [T0 + hours(1.49)], T0, [1]],
        [T0 + hours(0.9), [T0 + hours(1.51)], T0, [2]],
        [T0, [T1], T0, [1]],
        #
        [T0, [T0, T1], T0, [0, 1]],
        [
            T0 + hours(0.51),
            [
                T0 + hours(0.51),
                T0 + hours(0.52),
                T1,
                T1 + hours(0.49),
                T1 + hours(0.51),
            ],
            T0,
            [1, 1, 1, 1, 2],
        ],
    ],
)
def test_nwp_data_source_check_times_many_steps(
    now, ts, expected_time, expected_steps, nwp_data_sources
):
    data = nwp_data_sources.get(now=now, timestamps=ts)

    # Always one init_tie, one step, 3 variables, and 3x3 lat/lon.
    assert data.size == 3 * 3 * 3 * len(expected_steps)
    assert to_pydatetime(data.coords["time"].values) == expected_time
    assert_array_equal(data.coords["step"].values, [np.timedelta64(s, "h") for s in expected_steps])


@pytest.mark.parametrize("reverse_x", [True, False])
@pytest.mark.parametrize("reverse_y", [True, False])
@pytest.mark.parametrize(
    "lat,lon,expected_size",
    [
        [(LAT0, LAT2), (LON0, LON2), 27],
        [(LAT0, LAT1), (LON0, LON2), 3 * 2 * 3],
        [(LAT0, LAT1), (LON0, LON1), 3 * 2 * 2],
        [(LAT1, LAT1), (LON1, LON1), 3 * 1 * 1],
        [(None, None), (None, None), 27],
    ],
)
def test_nwp_data_source_space(reverse_x, reverse_y, lat, lon, expected_size, nwp_data_sources):
    # Reverse the data order and see if our flag words.
    if reverse_x:
        nwp_data_sources._data = nwp_data_sources._data.sortby("x", ascending=False)
        nwp_data_sources._x_is_ascending = False
    if reverse_y:
        nwp_data_sources._data = nwp_data_sources._data.sortby("y", ascending=False)
        nwp_data_sources._y_is_ascending = False

    data = nwp_data_sources.get(
        now=T0,
        timestamps=T0,
        min_lat=lat[0],
        max_lat=lat[1],
        min_lon=lon[0],
        max_lon=lon[1],
    )

    assert data.size == expected_size


def test_nwp_data_source_nearest(nwp_data_sources):
    data = nwp_data_sources.get(
        now=T0, timestamps=[T0, T0 + hours(2)], nearest_lat=LAT1, nearest_lon=LON1
    )
    x, y = nwp_data_sources._coordinate_transformer([(LAT1, LON1)])[0]
    assert y == data.coords["y"]
    assert x == data.coords["x"]
    assert data.size == 6


def test_nwp_data_source_tolerance(nwp_data_sources):
    now = T1 + timedelta(days=2)
    data = nwp_data_sources.get(
        now=now,
        timestamps=now,
    )

    assert data is not None
    data = nwp_data_sources.get(now=now, timestamps=now, tolerance="24h")
    assert data is None
