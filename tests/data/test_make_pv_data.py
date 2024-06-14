import pandas as pd
import xarray as xr
import pytest
from unittest.mock import patch
from datetime import datetime
from quartz_solar_forecast.pydantic_models import PVSite

# Mock data for testing
mock_enphase_data_df = pd.DataFrame({
    'timestamp': [pd.Timestamp('2023-06-14 12:00:00'), pd.Timestamp('2023-06-14 12:30:00')],
    'power_kw': [8.7, 9.3]
})

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
def test_make_pv_data(mock_get_enphase, site, expected_data, ts=pd.Timestamp('2023-06-14 12:15:00')):
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

    print("Result:")
    print(result)
    print("Expected:")
    print(expected_xr)

    assert result.equals(expected_xr)

@pytest.mark.parametrize("site, expected_data", [
    (PVSite(latitude=40.7128, longitude=-74.0059, capacity_kwp=8.5, inverter_type='enphase'), pd.DataFrame({'timestamp': [], 'power_kw': []})),
])
@patch('quartz_solar_forecast.inverters.enphase.get_enphase_data', return_value=pd.DataFrame({'timestamp': [], 'power_kw': []}))
def test_make_pv_data_empty_data(mock_get_enphase, site, expected_data, ts=pd.Timestamp('2023-06-14 12:15:00')):
    from quartz_solar_forecast.data import make_pv_data
    result = make_pv_data(site, ts)
    if not expected_data.empty:
        expected = expected_data[expected_data['timestamp'] <= ts]
    else:
        expected = expected_data
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

    print("Result (Empty Data):")
    print(result)
    print("Expected (Empty Data):")
    print(expected_xr)

    assert result.equals(expected_xr)
