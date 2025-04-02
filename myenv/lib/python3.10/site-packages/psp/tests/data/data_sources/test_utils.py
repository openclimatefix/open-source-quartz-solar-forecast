from psp.data_sources.utils import slice_on_lat_lon
from psp.gis import CoordinateTransformer


def test_slice_on_max_min(nwp_data_sources):

    transformer = CoordinateTransformer(from_=4326, to=27700)

    data = nwp_data_sources["UKV"]._data
    print(data)
    new_data = slice_on_lat_lon(
        data=data,
        max_lat=55,
        min_lat=48,
        max_lon=-1,
        min_lon=-4,
        transformer=transformer,
        y_is_ascending=False,
        x_is_ascending=True,
    )

    assert new_data.x.size == 2
    assert new_data.y.size == 2


def test_slice_on_nearest(nwp_data_sources):

    transformer = CoordinateTransformer(from_=4326, to=27700)

    data = nwp_data_sources["UKV"]._data
    new_data = slice_on_lat_lon(
        data=data,
        nearest_lat=52,
        nearest_lon=-2,
        transformer=transformer,
        y_is_ascending=False,
        x_is_ascending=True,
    )

    assert new_data.x.size == 1
    assert new_data.y.size == 1
