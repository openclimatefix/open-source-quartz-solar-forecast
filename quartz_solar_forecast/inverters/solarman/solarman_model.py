from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

from quartz_solar_forecast.inverters.inverter_model import AbstractInverter
from quartz_solar_forecast.inverters.solarman.solarman import get_solarman_data


class SolarmanSettings(BaseSettings):
    url: str = Field(alias="SOLARMAN_API_URL")
    token: str = Field(alias="SOLARMAN_TOKEN")
    id: str = Field(alias="SOLARMAN_ID")


class SolarmanInverter(AbstractInverter):

    def __init__(self, settings: SolarmanSettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=1)
            solarman_data = get_solarman_data(start_date, end_date, self.__settings)

            # Filter out rows with null power_kw values
            valid_data = solarman_data.dropna(subset=['power_kw'])

            if valid_data.empty:
                print("No valid Solarman data found.")
                return pd.DataFrame(columns=['timestamp', 'power_kw'])

            return valid_data
        except Exception as e:
            print(f"Error retrieving Solarman data: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'power_kw'])
