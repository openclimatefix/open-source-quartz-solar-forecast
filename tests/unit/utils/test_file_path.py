from quartz_solar_forecast.utils.file_path import get_file_path
from datetime import datetime


def test_get_file_path():
    latitude = 51.75
    longitude = -1.25
    capacity_kwp = 1.25
    date = datetime(2024, 7, 26, 12, 0, 0)
    path = get_file_path(latitude, longitude, capacity_kwp, "gb", date)

    assert path == "data/2024/7/26/gb_51.75_-1.25_1.25_20240726_12.csv"
