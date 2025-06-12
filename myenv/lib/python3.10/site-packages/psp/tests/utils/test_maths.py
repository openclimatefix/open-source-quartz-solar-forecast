import numpy as np
import pytest
import xarray as xr

from psp.utils.maths import safe_div


def assert_equal(a, b):
    assert type(a) == type(b)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "num,den,fallback,expected",
    [
        [1.0, 2.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 3.0, 3.0],
        [1.0, 0.0, np.nan, np.nan],
        [1.0, 0.0, None, 0.0],
        [0.0, 0.0, None, 0.0],
        [0.0, 0.0, np.nan, np.nan],
        [np.nan, np.nan, None, 0.0],
        [np.inf, np.nan, None, 0.0],
        [np.nan, np.inf, None, 0.0],
        [np.nan, 0.0, None, 0.0],
        [0.0, 0.0, None, 0.0],
        [np.nan, 0.0, None, 0.0],
        [0.0, np.nan, None, 0.0],
        #
        [
            np.array([[0.0, 1.0], [np.nan, np.inf]]),
            np.array([[0.0, np.nan], [0.0, np.nan]]),
            None,
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ],
        #
        [
            xr.DataArray([[0.0, 1], [2, 3]]),
            xr.DataArray([[0.0, 1], [2, 3]]),
            None,
            xr.DataArray([[0, 1], [1, 1]]),
        ],
        #
        [
            xr.DataArray([[0.0, 1], [2, 3]]),
            np.array([[0.0, 1], [2, 3]]),
            None,
            xr.DataArray([[0, 1], [1, 1]]),
        ],
        [
            xr.DataArray([0.0, 1, 2, 3]),
            np.array([0.0, 1, 2, 3]),
            None,
            xr.DataArray([0, 1, 1, 1]),
        ],
    ],
)
def test_safe_div(num, den, fallback, expected):
    if fallback is None:
        # Treat the special `nan` case.
        assert_equal(safe_div(num, den), expected)
    else:
        assert_equal(safe_div(num, den, fallback), expected)
