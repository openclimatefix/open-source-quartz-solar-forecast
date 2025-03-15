## NWP data

The nwp data has been generated with:

```python
import datetime as dt

import ocf_blosc2
import xarray as xr

from psp.data_sources.utils import slice_on_lat_lon
from psp.gis import CoordinateTransformer

nwp = xr.open_dataset(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
    "/UK_Met_Office/UKV/zarr/UKV_2020_NWP.zarr"
)

nwp = slice_on_lat_lon(
    nwp,
    min_lat=51.1,
    max_lat=53.7,
    min_lon=-3.3,
    max_lon=-2.7,
    transformer=CoordinateTransformer(4326, 27700),
    x_is_ascending=True,
    y_is_ascending=False,
)

nwp = nwp.sel(variable=["dswrf", "lcc", "mcc", "hcc"])

nwp = nwp.sel(init_time=slice(dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 14)))

nwp = nwp.isel(x=[0, -1], y=[0, -1])

nwp = nwp.sel(step=slice(dt.timedelta(hours=0), dt.timedelta(hours=4)))

for k in list(nwp['UKV'].attrs):
    del nwp['UKV'].attrs[k]

chunks = {
    "init_time": -1,
    "x": -1,
    "y": -1,
    "step": -1,
    "variable": -1,
}

nwp.to_zarr(
    "psp/tests/fixtures/nwp.zarr",
    mode="w",
    encoding={"UKV": {"chunks": [chunks[d] for d in nwp.dims]}},
)
```

The satellite data has been generated with
```python
import datetime as dt

import ocf_blosc2
import xarray as xr
from psp.data_sources.satellite import SatelliteDataSource
from psp.data_sources.utils import slice_on_lat_lon
from psp.gis import CoordinateTransformer

paths = [
    "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
]

sat = SatelliteDataSource(paths_or_data=paths, x_is_ascending=False)

x_min, y_min = sat.lonlat_to_geostationary(xx=-3.3, yy=51.1)
x_max, y_max = sat.lonlat_to_geostationary(xx=-2.7, yy=53.7)
d = sat._data
sat = sat._data


sat = sat.sel(variable=["IR_016", "IR_039", "IR_087", "IR_097"])

sat = sat.sel(x=slice(x_max, x_min))
sat = sat.sel(y=slice(y_min, y_max))
sat = sat.sel(time=slice(dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 14)))


# rename back to old variables
sat = sat.rename({'x':'x_geostationary', 'y':'y_geostationary', 'value':'data'})

chunks = {
    "time": -1,
    "x_geostationary": -1,
    "y_geostationary": -1,
    "variable": -1,
}

sat.to_zarr(
    "psp/tests/fixtures/satellite.zarr",
    mode="w",
    encoding={"data": {"chunks": [chunks[d] for d in sat.dims]}},
)
```