from datetime import datetime
from typing import List

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry


class WeatherService:
    def __init__(self):
        """
        Initialize the WeatherService.

        This class provides high-level weather-related functionality using OpenMeteo API.
        """
        pass

    def _build_url(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: List[str],
    ) -> str:
        """
        Build the URL for the OpenMeteo API.

        Parameters
        ----------
        latitude : float
            The latitude of the location for which to get weather data.
        longitude : float
            The longitude of the location for which to get weather data.
        start_date : str
            The start date for the weather data, in the format YYYY-MM-DD.
        end_date : str
            The end date for the weather data, in the format YYYY-MM-DD.
        variables : list
            A list of weather variables to include in the API response.

        Returns
        -------
        str
            The URL for the OpenMeteo API.
        """
        url = "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly={variables}&start_date={start_date}&end_date={end_date}&timezone=GMT".format(
            latitude=latitude,
            longitude=longitude,
            variables=",".join(variables),
            start_date=start_date,
            end_date=end_date,
        )
        return url

    def _validate_coordinates(self, latitude: float, longitude: float) -> None:
        """
        Validate latitude and longitude coordinates.

        Parameters
        ----------
        latitude : float
            The latitude value to be checked.
        longitude : float
            The longitude value to be checked.

        Raises
        ------
        ValueError
            If coordinates are not within valid ranges.
        """
        assert (
            -90 <= latitude <= 90 and -180 <= longitude <= 180
        ), "Invalid coordinates. Latitude must be between -90 and 90, and longitude must be between -180 and 180."

    def _validate_date_format(self, start_date: str, end_date: str) -> None:
        """
        Validate date format and check if end_date is greater than start_date.

        Parameters
        ----------
        start_date : str
            Start date in format YYYY-MM-DD.
        end_date : str
            End date in format YYYY-MM-DD.

        Raises
        ------
        ValueError
            If date format is invalid or end_date is not greater than start_date.
        """
        try:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            assert end_datetime > start_datetime, "End date must be greater than start date."
        except (ValueError, AssertionError) as e:
            raise ValueError(
                f"Invalid date format or range. Please use YYYY-MM-DD and ensure end_date is greater than start_date. Error: {str(e)}"
            )

    def get_hourly_weather(
        self, latitude: float, longitude: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get hourly weather data ranging from 3 months ago up to 15 days ahead (forecast).

        Parameters
        ----------
        latitude : float
            The latitude of the location for which to get weather data.
        longitude : float
            The longitude of the location for which to get weather data.
        start_date : str
            The start date for the weather data, in the format YYYY-MM-DD.
        end_date : str
            The end date for the weather data, in the format YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the hourly weather data for the specified location and date range.

        Raises
        ------
        ValueError
            If the provided coordinates are invalid or if the date format is invalid.
        """
        self._validate_coordinates(latitude, longitude)
        self._validate_date_format(start_date, end_date)

        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "is_day",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation",
        ]
        url = self._build_url(latitude, longitude, start_date, end_date, variables)

        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        try:
            openmeteo = openmeteo_requests.Client(session=retry_session)
            response = openmeteo.weather_api(url, params={})
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to OpenMeteo API timed out. URl - {url}")

        hourly = response[0].Hourly()
        hourly_data = {"time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=False),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=False),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        for i, variable in enumerate(variables):
            hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(hourly_data)
        df["time"] = pd.to_datetime(df["time"])

        # rename time column to date
        df = df.rename(
            columns={
                "time": "date",
            }
        )

        return df
