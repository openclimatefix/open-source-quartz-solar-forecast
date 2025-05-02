import numpy as np
import pytest
from numpy.testing import assert_allclose

from psp.gis import CoordinateTransformer, approx_add_meters_to_lat_lon


@pytest.mark.parametrize(
    "p0",
    [
        np.array([[50, 0], [45.543526844263546, -73.5694752214446]]),
        np.array([1, 1]),
        [1, 1],
    ],
)
def test_approx_add_meters_to_lat_lon_happy_path(p0):
    p1 = approx_add_meters_to_lat_lon(p0, [0, 10])
    p2 = approx_add_meters_to_lat_lon(p1, [-10, 0])
    p3 = approx_add_meters_to_lat_lon(p2, [10, -10])

    assert_allclose(p0, p3, atol=1e-10)

    # Weird but should also work!
    p1 = approx_add_meters_to_lat_lon(p0, p0)
    p2 = approx_add_meters_to_lat_lon(p1, -np.array(p0))

    assert_allclose(p0, p2, atol=1e-10)


@pytest.mark.parametrize(
    "coord1,coord2,input_points,expected_points,",
    [
        [
            27700,
            4326,
            [
                (622575.7031043093, -5527063.8148287395),
                (3964428.6092978613, 1425253.705593198),
            ],
            [(0.0, 0.0), (50.0, 50.0)],
        ],
        [4326, 4326, [(10, 11), (12, 13)], [(10, 11), (12, 13)]],
    ],
)
def test_coordinate_transformer(coord1, coord2, input_points, expected_points):
    transform = CoordinateTransformer(coord1, coord2)

    new_points = transform(input_points)
    assert len(new_points) == len(expected_points)
    for p1, p2 in zip(expected_points, new_points):
        assert_allclose(p1, p2, atol=1e-10)


def test_coordinate_transformer_lat_lon_lat_lon():
    transform = CoordinateTransformer(4326, 4326)

    points = [(50, -1), (54, 1)]

    new_points = transform(points)

    assert len(new_points) == len(points)
    for p1, p2 in zip(points, new_points):
        assert_allclose(p1, p2, atol=1e-10)

    transform = CoordinateTransformer(27700, 4326)
    new_points = transform(points)
    transform = CoordinateTransformer(4326, 27700)
