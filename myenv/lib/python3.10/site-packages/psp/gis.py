from math import cos, radians, sqrt
from typing import Iterable, Sequence

import numpy as np
import pyproj

# Radius of the earth in meters.
EARTH_RADIUS = 6371_000


def approx_distance(lat_lon1: Sequence[float], lat_lon2: Sequence[float]) -> float:
    """Return the approximate distance between two (lat,lon) coodinates.

    This works if they are not too far (we ignore the curvature of the earth).
    The points are expected to be in degrees and the distance is returned in meters.
    """
    lat1, lon1 = lat_lon1
    lat2, lon2 = lat_lon2

    lat1, lon1, lat2, lon2 = [radians(x) for x in [lat1, lon1, lat2, lon2]]

    lat = (lat1 + lat2) / 2

    dy = EARTH_RADIUS * (lat2 - lat1)
    dx = EARTH_RADIUS * (lon2 - lon1) * cos(lat)

    return sqrt(dx**2 + dy**2)


def _parse_list_of_points(x: np.ndarray | list[float]) -> tuple[bool, np.ndarray]:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # Make sure we have an (n, 2)-shaped array.
    if len(x.shape) == 1:
        was_1d = True
        x = np.atleast_2d(x)
    else:
        was_1d = False

    # Make mypy happy.
    assert isinstance(x, np.ndarray)

    assert len(x.shape) == 2
    assert x.shape[1] == 2

    return was_1d, x


def approx_add_meters_to_lat_lon(
    lat_lon: np.ndarray | list[float],
    delta_meters: np.ndarray | list[float],
) -> np.ndarray:
    """Approximately add [y, x] meters to a matrix of [lat, lon].

    We assume that earth is a perfect sphere and that displacements are small enough as for the
    earth to feel flat.
    This should be fine for small values of `meters`.
    """
    was_1d_1, lat_lon = _parse_list_of_points(lat_lon)
    was_1d_2, delta_meters = _parse_list_of_points(delta_meters)

    assert isinstance(lat_lon, np.ndarray)

    lat_rad = np.radians(lat_lon[:, 1]).reshape(-1, 1)

    cos_lat = np.cos(lat_rad)

    delta_x_rad = delta_meters[:, 1].reshape(-1, 1) / EARTH_RADIUS / cos_lat
    delta_y_rad = delta_meters[:, 0].reshape(-1, 1) / EARTH_RADIUS

    delta_x = np.degrees(delta_x_rad)
    delta_y = np.degrees(delta_y_rad) * np.ones_like(delta_x)

    delta = np.hstack([delta_y, delta_x])

    out = lat_lon + delta

    if was_1d_1 and was_1d_2:
        out = out[0]

    return out


Point = tuple[float, float]


class CoordinateTransformer:
    """Convenience wrapper around pyproj's Transformer API.

    Example:
    -------
    >>> transform = CoordinateTransformer(27700, 4326)
    >>> transform([(1000., 2000.), (3000., 4000.)])
    """

    def __init__(self, from_: int, to: int):
        """Define the coordinate systems from and to which we want to transform.

        Arguments:
        ---------
        from_: Input coordinate reference system code.
        to: Output coordinate reference system code.
        """
        self._transformer = pyproj.Transformer.from_crs(from_, to)

    def __call__(self, points: Iterable[Point]) -> list[Point]:
        """Transform `points` from one coordinate system to the other."""
        return list(self._transformer.itransform(points))  # type: ignore
