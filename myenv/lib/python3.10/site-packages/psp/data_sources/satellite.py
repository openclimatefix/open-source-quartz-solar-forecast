import pyresample
import xarray as xr

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.utils import _TIME, _VALUE, _VARIABLE, _X, _Y
from psp.gis import CoordinateTransformer


class SatelliteDataSource(NwpDataSource):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            filter_on_step=False,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            value_name="data",
        )

        # Get the coordinate transformer.# get crs
        area_definition_yaml = self._data.value.attrs["area"]
        geostationary_area_definition = pyresample.area_config.load_area_from_string(
            area_definition_yaml
        )
        geostationary_crs = geostationary_area_definition.crs

        # Get the coordinate transformer, from lat/lon to geostationary.
        self._coordinate_transformer = CoordinateTransformer(from_=4326, to=geostationary_crs)

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        # Rename the dimensions.
        rename_map: dict[str, str] = {}
        for old, new in zip(
            [
                self._x_dim_name,
                self._y_dim_name,
                self._time_dim_name,
                self._variable_dim_name,
                self._value_name,
            ],
            [_X, _Y, _TIME, _VARIABLE, _VALUE],
        ):
            if old != new:
                rename_map[old] = new

        data = data.rename(rename_map)

        # Filter data to keep only the variables in self._nwp_variables if it's not None
        if self._variables is not None:
            data = data.sel(variable=self._variables)

        return data

    def _open(self, paths: list[str]) -> xr.Dataset:
        d = xr.open_mfdataset(
            paths,
            engine="zarr",
            concat_dim="time",
            combine="nested",
            chunks="auto",
            join="override",
        )
        return d
