from typing import TypeVar

import xarray as xr

from psp.gis import CoordinateTransformer

_X = "x"
_Y = "y"
_TIME = "time"
_STEP = "step"
_VARIABLE = "variable"
_VALUE = "value"

T = TypeVar("T", bound=xr.Dataset | xr.DataArray)


def slice_on_lat_lon(
    data: T,
    *,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    nearest_lat: float | None = None,
    nearest_lon: float | None = None,
    transformer: CoordinateTransformer,
    x_is_ascending: bool,
    y_is_ascending: bool,
) -> T:
    """
    Slice the data on lat/lon

    Args:
    ----
        data: The data to slice
        min_lat: The minimum latitude to slice on
        max_lat: The maximum latitude to slice on
        min_lon: The minimum longitude to slice on
        max_lon: The maximum longitude to slice on
        nearest_lat: The latitude to slice on
        nearest_lon: The longitude to slice on
        transformer: The transformer to use to convert lat/lon to x/y
        x_is_ascending: Whether the x values are ascending
        y_is_ascending: Whether the y values are ascending
        do_average: Take average over the area, of x and y coordinates
    """
    # Only allow `None` values for lat/lon if they are all None (in which case we don't filter
    # by lat/lon).
    num_none = sum([x is None for x in [min_lat, max_lat, min_lon, max_lon]])
    assert num_none in [0, 4]

    if min_lat is not None:
        assert min_lat is not None
        assert min_lon is not None
        assert max_lat is not None
        assert max_lon is not None

        assert max_lat >= min_lat
        assert max_lon >= min_lon

        # This looks funny because when going from lat/lon to osgb we have to use
        # (x, y) = transformer([(lat, lon)])
        # however for lat/lon to geostationary we have to use
        # (x_geo, y_geo) = transformer([(lon, lat)])

        points = [(min_lat, min_lon), (max_lat, max_lon)]
        point1, point2 = transformer(points)
        min_x, min_y = point1
        max_x, max_y = point2

        if not x_is_ascending:
            min_x, max_x = max_x, min_x
        if not y_is_ascending:
            min_y, max_y = max_y, min_y

        new_data = data.sel(x=slice(min_x, max_x), y=slice(min_y, max_y))

        # Type ignore because this is still simpler than adding some `@overload`.
        return new_data  # type: ignore

    elif nearest_lat is not None and nearest_lon is not None:
        ((x, y),) = transformer([(nearest_lat, nearest_lon)])

        return data.sel(x=x, y=y, method="nearest")  # type: ignore

    return data
