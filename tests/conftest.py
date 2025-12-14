"""
Pytest configuration for mocking Open-Meteo API calls.
This eliminates 20-minute wait times in CI by mocking external API requests.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest


# Check if we're in CI environment
IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("PYTEST_CURRENT_TEST") is not None


class MockOpenMeteoResponse:
    """Mock response object that matches openmeteo_requests response structure."""

    def __init__(self, hourly_data):
        self._hourly_data = hourly_data

    def Hourly(self):
        """Return mock hourly data object."""
        mock_hourly = Mock()

        # Set up time range
        start_time = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(hours=len(self._hourly_data["temperature_2m"]))).timestamp()

        mock_hourly.Time.return_value = int(start_time)
        mock_hourly.TimeEnd.return_value = int(end_time)
        mock_hourly.Interval.return_value = 3600  # 1 hour in seconds

        # Mock the Variables method to return data for each variable
        def mock_variables(index):
            variable_mock = Mock()
            variable_names = list(self._hourly_data.keys())
            if index < len(variable_names):
                import numpy as np
                variable_mock.ValuesAsNumpy.return_value = np.array(self._hourly_data[variable_names[index]])
            return variable_mock

        mock_hourly.Variables = mock_variables

        return mock_hourly


def generate_mock_weather_data(hours=48, is_historical=False):
    """Generate realistic mock weather data."""
    base_temp = 12.0 if is_historical else 15.0

    data = {
        "temperature_2m": [base_temp + i * 0.1 for i in range(hours)],
        "relative_humidity_2m": [70.0 - i * 0.1 for i in range(hours)],
        "dew_point_2m": [10.0 + i * 0.05 for i in range(hours)],
        "precipitation": [0.0] * hours,
        "surface_pressure": [1013.0 + i * 0.1 for i in range(hours)],
        "cloud_cover": [30.0 + i * 0.3 for i in range(hours)],
        "cloud_cover_low": [20.0 + i * 0.2 for i in range(hours)],
        "cloud_cover_mid": [15.0 + i * 0.15 for i in range(hours)],
        "cloud_cover_high": [10.0 + i * 0.1 for i in range(hours)],
        "wind_speed_10m": [5.0 + i * 0.1 for i in range(hours)],
        "wind_direction_10m": [180.0 + i for i in range(hours)],
        "is_day": [1 if 6 <= (i % 24) <= 20 else 0 for i in range(hours)],
        "shortwave_radiation": [0 if (i % 24) < 6 or (i % 24) > 20 else 300 + (i % 24) * 20 for i in range(hours)],
        "direct_radiation": [0 if (i % 24) < 6 or (i % 24) > 20 else 100 + (i % 24) * 10 for i in range(hours)],
        "diffuse_radiation": [0 if (i % 24) < 6 or (i % 24) > 20 else 50 + (i % 24) * 5 for i in range(hours)],
        "direct_normal_irradiance": [0 if (i % 24) < 6 or (i % 24) > 20 else 200 + (i % 24) * 15 for i in range(hours)],
        "terrestrial_radiation": [50.0 + i * 0.2 for i in range(hours)],
    }

    return data


@pytest.fixture(scope="function")
def mock_weather_api(monkeypatch):
    """
    Mock the openmeteo_requests.Client.weather_api method.
    Only activates in CI or when MOCK_WEATHER_API env var is set.
    """

    # Only mock in CI or when explicitly requested
    should_mock = IS_CI or os.getenv("MOCK_WEATHER_API", "false").lower() == "true"

    if not should_mock:
        yield
        return

    original_weather_api = None

    def mock_weather_api_call(url, params=None):
        """Mock implementation of weather_api call."""
        # Determine if this is a historical request based on date in URL
        is_historical = False
        if "start_date" in url:
            import re
            date_match = re.search(r"start_date=(\d{4}-\d{2}-\d{2})", url)
            if date_match:
                start_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                # Historical if more than 180 days ago
                is_historical = (datetime.now() - start_date).days > 180

        # Generate appropriate mock data
        hours = 168 if is_historical else 48  # 7 days vs 2 days
        hourly_data = generate_mock_weather_data(hours=hours, is_historical=is_historical)

        # Return list with single mock response (API returns list)
        return [MockOpenMeteoResponse(hourly_data)]

    # Patch the weather_api method
    with patch('openmeteo_requests.Client.weather_api', side_effect=mock_weather_api_call):
        yield


# Auto-apply the mock to all tests when in CI
@pytest.fixture(autouse=True)
def auto_mock_in_ci(mock_weather_api):
    """Automatically apply weather API mocking in CI environment."""
    yield
