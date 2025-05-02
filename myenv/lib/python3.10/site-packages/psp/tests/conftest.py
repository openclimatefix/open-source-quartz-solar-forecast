import pytest

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource
from psp.data_sources.satellite import SatelliteDataSource


@pytest.fixture
def pv_data_source():
    return NetcdfPvDataSource(
        "psp/tests/fixtures/pv_data.nc",
        id_dim_name="ss_id",
        timestamp_dim_name="timestamp",
        rename={"generation_wh": "power"},
    )


@pytest.fixture
def nwp_data_source(pv_data_source):
    return NwpDataSource(
        "psp/tests/fixtures/nwp.zarr",
        coord_system=27700,
        time_dim_name="init_time",
        value_name="UKV",
        y_is_ascending=False,
    )


@pytest.fixture
def nwp_data_sources(pv_data_source):
    return {
        "UKV": NwpDataSource(
            "psp/tests/fixtures/nwp.zarr",
            coord_system=27700,
            time_dim_name="init_time",
            value_name="UKV",
            y_is_ascending=False,
        ),
    }


@pytest.fixture
def satellite_data_sources(pv_data_source):
    return {
        "EUMETSAT": SatelliteDataSource(
            "psp/tests/fixtures/satellite.zarr",
            x_is_ascending=False,
        ),
    }
