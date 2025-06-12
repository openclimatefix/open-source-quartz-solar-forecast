from datetime import datetime

import ocf_blosc2  # noqa
import xarray as xr

from psp.data_sources.satellite import SatelliteDataSource


def test_satellite_data_source():
    """Test loading the satellite data

    Note this test uses the satellite public dataset to get this data.
    It can take about 30 seconds to run

    """
    # this is for the google datasets
    paths = [
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
    ]

    sat = SatelliteDataSource(paths_or_data=paths, x_is_ascending=False)

    now = datetime(2021, 2, 1)
    lat = 50
    lon = 0

    example = sat.get(now=now, timestamps=[now], nearest_lat=lat, nearest_lon=lon)

    assert isinstance(example, xr.DataArray)
    assert example.x.size > 0
    assert example.y.size > 0

    example = sat.get(
        now=now, timestamps=[now], max_lat=lat + 1, min_lat=lat, max_lon=lon + 1, min_lon=lon
    )
    assert isinstance(example, xr.DataArray)
    assert example.x.size > 0
    assert example.y.size > 0
