import pandas as pd
import numpy as np
import xarray as xr
import pytest
from unittest.mock import patch
from datetime import datetime
from quartz_solar_forecast.pydantic_models import PVSite

def mock_enphase_data(*args, **kwargs):
    return pd.DataFrame({
        'timestamp': [
            datetime(2024, 6, 5, 11, 25),
            datetime(2024, 6, 5, 11, 30),
            datetime(2024, 6, 5, 11, 35)
        ],
        'power_kw': [0.5, 0.6, 0.7]
    })

@pytest.mark.parametrize("site, expected_data", [
    (PVSite(latitude=40.7128, longitude=-74.0059, capacity_kwp=8.5, inverter_type='enphase'), mock_enphase_data()),
])
@patch('quartz_solar_forecast.inverters.enphase.get_enphase_data', side_effect=mock_enphase_data)
def test_make_pv_data_enphase(mock_get_enphase, site, expected_data, ts=pd.Timestamp('2023-06-14 12:15:00')):
    from quartz_solar_forecast.data import make_pv_data
    result = make_pv_data(site, ts)
    expected = expected_data[expected_data['timestamp'] <= ts]
    expected_xr = xr.DataArray(
        data=expected['power_kw'].values.reshape(1, -1),
        dims=['pv_id', 'timestamp'],
        coords={
            'longitude': (['pv_id'], [site.longitude]),
            'latitude': (['pv_id'], [site.latitude]),
            'timestamp': (['timestamp'], expected['timestamp'].values.astype('datetime64[ns]')),
            'pv_id': [1],
            'kwp': (['pv_id'], [site.capacity_kwp]),
            'tilt': (["pv_id"], [site.tilt]),
            'orientation': (["pv_id"], [site.orientation]),
        }
    ).to_dataset(name='generation_kw')

    assert result.equals(expected_xr)
    
@pytest.mark.parametrize("site, expected_data", [
    (PVSite(latitude=40.7128, longitude=-74.0059, capacity_kwp=8.5, inverter_type='unknown'), np.array([[np.nan]])),
])
def test_make_pv_data_no_live(site, expected_data, ts=pd.Timestamp('2023-06-14 12:15:00')):
    from quartz_solar_forecast.data import make_pv_data
    result = make_pv_data(site, ts)
    expected_xr = xr.DataArray(
        data=expected_data,
        dims=['pv_id', 'timestamp'],
        coords={
            'longitude': (['pv_id'], [site.longitude]),
            'latitude': (['pv_id'], [site.latitude]),
            'timestamp': (['timestamp'], [ts]),
            'pv_id': [1],
            'kwp': (['pv_id'], [site.capacity_kwp]),
            'tilt': (["pv_id"], [site.tilt]),
            'orientation': (["pv_id"], [site.orientation]),
        }
    ).to_dataset(name='generation_kw')

    assert result.equals(expected_xr)