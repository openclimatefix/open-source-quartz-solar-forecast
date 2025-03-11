import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def mock_metadata():
    """Mock the metadata.csv from HuggingFace"""
    return pd.DataFrame({
        'capacity_kwp': [4.1, 3.8],
        'latitude': [51.75, 51.76], 
        'longitude': [-1.25, -1.26],
        'orientation': [180, 185],
        'tilt': [35, 40]
    })

@pytest.fixture
def mock_pv_data():
    """Mock the pv.netcdf data from HuggingFace"""
    # Create sample time series data
    times = pd.date_range(
        start=datetime.now(), 
        periods=24,
        freq='H'
    )
    return {
        'power_w': np.random.rand(24, 2) * 1000,  # 24 hours x 2 sites
        'time': times,
        'site_id': [0, 1]
    }

@pytest.fixture(autouse=True)
def skip_integration_tests(request):
    """Skip integration tests unless explicitly requested"""
    if request.node.get_closest_marker('integration') and not request.config.getoption('--run-integration'):
        pytest.skip('Integration test skipped. Use --run-integration to run.')

def pytest_addoption(parser):
    """Add command-line options for pytest"""
    parser.addoption(
        '--run-integration', 
        action='store_true', 
        default=False,
        help='Run integration tests that require external services'
    ) 