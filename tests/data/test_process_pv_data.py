import pytest
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from quartz_solar_forecast.data import process_pv_data  
from quartz_solar_forecast.pydantic_models import PVSite

@pytest.fixture
def sample_site():
    return PVSite(
        latitude=51.75,
        longitude=-1.25,
        capacity_kwp=1.25,
        tilt=35,
        orientation=180,
        inverter_type="enphase"
    )

@pytest.fixture
def sample_timestamp():
    timestamp = datetime.now().timestamp()
    timestamp_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return pd.to_datetime(timestamp_str)

@pytest.fixture
def sample_live_generation():
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2024-06-16 10:00:00'),
            pd.Timestamp('2024-06-16 10:05:00'),
            pd.Timestamp('2024-06-16 10:10:00')
        ],
        'power_kw': [0.75, 0.80, 0.78]
    })

def test_process_pv_data_with_live_data(sample_site, sample_timestamp, sample_live_generation):
    result = process_pv_data(sample_live_generation, sample_timestamp, sample_site)

    assert isinstance(result, xr.Dataset)
    assert 'generation_kw' in result.data_vars
    assert set(result.coords) == {'longitude', 'latitude', 'timestamp', 'pv_id', 'kwp', 'tilt', 'orientation'}
    assert result.pv_id.values.tolist() == [1]
    assert result.longitude.values.tolist() == [sample_site.longitude]
    assert result.latitude.values.tolist() == [sample_site.latitude]
    assert result.kwp.values.tolist() == [sample_site.capacity_kwp]
    assert result.tilt.values.tolist() == [sample_site.tilt]
    assert result.orientation.values.tolist() == [sample_site.orientation]
    assert len(result.timestamp) <= len(sample_live_generation)
    assert np.all(result.timestamp.values <= sample_timestamp)

def test_process_pv_data_without_live_data(sample_site, sample_timestamp):
    result = process_pv_data(None, sample_timestamp, sample_site)

    assert isinstance(result, xr.Dataset)
    assert 'generation_kw' in result.data_vars
    assert set(result.coords) == {'longitude', 'latitude', 'timestamp', 'pv_id', 'kwp', 'tilt', 'orientation'}
    assert result.pv_id.values.tolist() == [1]
    assert result.longitude.values.tolist() == [sample_site.longitude]
    assert result.latitude.values.tolist() == [sample_site.latitude]
    assert result.kwp.values.tolist() == [sample_site.capacity_kwp]
    assert result.tilt.values.tolist() == [sample_site.tilt]
    assert result.orientation.values.tolist() == [sample_site.orientation]
    assert len(result.timestamp) == 1
    assert result.timestamp.values[0] == sample_timestamp
    assert np.isnan(result.generation_kw.values[0][0])