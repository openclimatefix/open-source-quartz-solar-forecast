import pytest
import pandas as pd
import numpy as np
from quartz_solar_forecast.inverters.enphase import process_enphase_data 

@pytest.fixture
def sample_data():
    return {
        'system_id': 3136663,
        'granularity': 'week',
        'total_devices': 4,
        'start_at': 1718530896,
        'end_at': 1719134971,
        'items': 'intervals',
        'intervals': [
            {'end_at': 1718531100, 'devices_reporting': 4, 'powr': 624, 'enwh': 52},
            {'end_at': 1718531400, 'devices_reporting': 4, 'powr': 684, 'enwh': 57},
            {'end_at': 1718531700, 'devices_reporting': 4, 'powr': 672, 'enwh': 56},
        ]
    }

def test_process_enphase_data(sample_data):
    # Set start_at to before/after the first interval
    start_at = sample_data['intervals'][0]['end_at'] + 1 
    
    # Process the data
    result = process_enphase_data(sample_data, start_at)
    
    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check if the DataFrame has the expected columns
    assert set(result.columns) == {'timestamp', 'power_kw'}
    
    # Check if the timestamp column is of datetime type
    assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    # Check if power_kw values are correctly calculated (divided by 1000)
    expected_power_values = [interval['powr'] / 1000 for interval in sample_data['intervals']]
    assert all(value in expected_power_values for value in result['power_kw'])
    
    # Convert start_at to a naive UTC timestamp
    start_at_timestamp = pd.Timestamp(start_at, unit='s').tz_localize('UTC').tz_convert(None)
    
    # Check if all timestamps are after the start_at time
    assert np.all(result['timestamp'] >= start_at_timestamp)
    
    # Check if the number of rows is less than or equal to the number of intervals
    assert len(result) <= len(sample_data['intervals'])

    # Check if timestamps are formatted correctly
    expected_timestamps = [
        pd.Timestamp(interval['end_at'], unit='s').tz_localize('UTC').tz_convert(None).strftime('%Y-%m-%d %H:%M:%S')
        for interval in sample_data['intervals']
    ]
    assert all(ts.strftime('%Y-%m-%d %H:%M:%S') in expected_timestamps for ts in result['timestamp'])