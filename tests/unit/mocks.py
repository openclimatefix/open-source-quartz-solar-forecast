import pandas as pd
import pytest

from quartz_solar_forecast.weather.open_meteo import WeatherService


# Fixture for getting hourly data from Open-Meteo API
@pytest.fixture
def mock_weather_api(monkeypatch):
    # Monkeypatch get_hourly_weather method:
    # behavior same to original method, returns dummy weather data (zeroes)
    def mock_get_hourly_weather(self, latitude, longitude, start_date, end_date, variables=[], api_type="forecast", model=None):
        mock_hourly_date = pd.date_range(
        	start = pd.to_datetime(start_date, format="%Y-%m-%d", utc = False),
        	end = pd.to_datetime(end_date, format="%Y-%m-%d", utc = False) + pd.Timedelta(days=1),
        	freq = pd.Timedelta(hours=1),
        	inclusive = "left"
        )
        mock_weather_df = pd.DataFrame({ "date": mock_hourly_date })
        # Fill with zeroes (fake weather data)
        for v in variables:
            mock_weather_df[v] = 0.0
        return mock_weather_df

    monkeypatch.setattr(WeatherService, "get_hourly_weather", mock_get_hourly_weather)
