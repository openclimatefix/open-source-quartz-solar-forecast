from typing import Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

from quartz_solar_forecast.inverters.givenergy.givenergy import get_givenergy_data
from quartz_solar_forecast.inverters.inverter_model import AbstractInverter


class GivEnergySettings(BaseSettings):
    api_key: str = Field(alias="GIVENERGY_API_KEY")


class GivEnergyInverter(AbstractInverter):

    def __init__(self, settings: GivEnergySettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            return get_givenergy_data(self.__settings)
        except Exception as e:
            print(f"Error retrieving GivEnergy data: {e}")
            return None
