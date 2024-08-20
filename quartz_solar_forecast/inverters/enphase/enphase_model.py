from typing import Optional

import pandas as pd

from quartz_solar_forecast.inverters.enphase.enphase import get_enphase_data
from quartz_solar_forecast.inverters.inverter_model import AbstractInverter
from pydantic import Field
from pydantic_settings import BaseSettings


class EnphaseSettings(BaseSettings):
    client_id: str = Field(alias="ENPHASE_CLIENT_ID")
    system_id: str = Field(alias="ENPHASE_SYSTEM_ID")
    api_key: str = Field(alias="ENPHASE_API_KEY")
    client_secret: str = Field(alias="ENPHASE_CLIENT_SECRET")


class EnphaseInverter(AbstractInverter):

    def __init__(self, settings: EnphaseSettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        return get_enphase_data(self.__settings)

