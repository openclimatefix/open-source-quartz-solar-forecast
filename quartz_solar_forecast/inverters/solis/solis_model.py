from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

from quartz_solar_forecast.inverters.inverter_model import AbstractInverter
from quartz_solar_forecast.inverters.solis.solis import get_solis_data


class SolisSettings(BaseSettings):
    api_url: str = Field(alias="SOLIS_CLOUD_API_URL", default='https://www.soliscloud.com')
    port: str = Field(alias="SOLIS_CLOUD_API_PORT", default='13333')
    api_key: str = Field(alias="SOLIS_CLOUD_API_KEY")
    client_secret: str = Field(alias="SOLIS_CLOUD_API_KEY_SECRET")


class SolisInverter(AbstractInverter):
    def __init__(self, settings: SolisSettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            return asyncio.run(get_solis_data(self.__settings))
        except Exception as e:
            print(f"Error retrieving Solis data: {str(e)}")
            return None
