import http.client
import pandas as pd
import numpy as np
import json
import xarray as xr
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from quartz_solar_forecast.pydantic_models import PVSite

def mock_api_response():
    mock_data = {
        'system_id': 3136663,
        'granularity': 'week',
        'total_devices': 4,
        'start_at': 1718174469,
        'end_at': 1718778676,
        'items': 'intervals',
        'intervals': [
            {'end_at': 1718655900, 'devices_reporting': 4, 'powr': 500, 'enwh': 0},
            {'end_at': 1718656200, 'devices_reporting': 4, 'powr': 600, 'enwh': 0},
            {'end_at': 1718656500, 'devices_reporting': 4, 'powr': 700, 'enwh': 0},
            # Add more intervals as needed
        ]
    }
    return mock_data

@pytest.fixture
def mock_get_enphase_auth_code(monkeypatch):
    def mock_auth_code(*args, **kwargs):
        return "mock_auth_code"

    monkeypatch.setattr("quartz_solar_forecast.inverters.enphase.get_enphase_authorization_code", mock_auth_code)

@pytest.fixture
def mock_get_enphase(monkeypatch):
    # Mock the get_enphase_access_token function
    monkeypatch.setattr('quartz_solar_forecast.inverters.enphase.get_enphase_access_token', lambda: "mock_access_token")

    # Mock the API response from get_enphase_data
    mock_api_response_data = mock_api_response()
    mock_api_response_json = json.dumps(mock_api_response_data).encode('utf-8')

    def mock_request_get(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.read.return_value = mock_api_response_json
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn.sock = MagicMock()  # Add a mock socket
        mock_conn._HTTPConnection__state = http.client._CS_REQ_SENT  # Set the state to REQ_SENT
        return mock_conn

    monkeypatch.setattr('http.client.HTTPSConnection', mock_request_get)

@pytest.mark.parametrize("site, expected_data", [
    (PVSite(latitude=40.7128, longitude=-74.0059, capacity_kwp=8.5, inverter_type='enphase'), mock_api_response()['intervals']),
])
def test_make_pv_data_enphase(mock_get_enphase, mock_get_enphase_auth_code, site, expected_data, ts=pd.Timestamp('2023-06-19 12:15:00')):
    from quartz_solar_forecast.data import make_pv_data
    result = make_pv_data(site, ts)
    expected_df = pd.DataFrame(expected_data)
    expected_df['end_at'] = expected_df['end_at'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    expected_df = expected_df.rename(columns={'end_at': 'timestamp', 'powr': 'power_kw'})
    ts_utc = pd.to_datetime(ts, utc=True)  # Convert ts to UTC datetime64[ns, UTC]
    expected = expected_df[expected_df['timestamp'] <= ts_utc]
    expected_xr = xr.DataArray(
        data=expected['power_kw'].values.reshape(1, -1) / 1000,
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
    (PVSite(latitude=40.7128, longitude=-74.0059, capacity_kwp=8.5), np.array([[np.nan]])),
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