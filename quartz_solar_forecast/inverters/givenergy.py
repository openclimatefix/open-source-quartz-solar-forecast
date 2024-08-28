from typing import Optional

import requests
import pandas as pd
from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from quartz_solar_forecast.inverters.inverter import AbstractInverter


class GivEnergySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

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


def get_inverter_serial_number(settings: GivEnergySettings):
    """
    Fetch the inverter serial number from the GivEnergy communication device API.
    
    :return: Inverter serial number as a string
    """
    api_key = settings.api_key
    
    if not api_key:
        raise ValueError("GIVENERGY_API_KEY not set in environment variables")

    url = 'https://api.givenergy.cloud/v1/communication-device'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Communication device API request failed with status code {response.status_code}")
    
    data = response.json()['data']
    if not data:
        raise ValueError("No communication devices found")
    
    inverter_serial_number = data[0]['inverter']['serial']
    return inverter_serial_number


def get_givenergy_data(settings: GivEnergySettings):
    """
    Fetch the latest data from the GivEnergy API and return a DataFrame.
    
    :return: DataFrame with timestamp and power_kw columns
    """
    api_key = settings.api_key
    
    if not api_key:
        raise ValueError("GIVENERGY_API_KEY not set in environment variables")

    inverter_serial_number = get_inverter_serial_number(settings)

    url = f'https://api.givenergy.cloud/v1/inverter/{inverter_serial_number}/system-data/latest'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"System data API request failed with status code {response.status_code}")
    
    data = response.json()['data']
    
    # Process the data
    timestamp = datetime.strptime(data['time'], "%Y-%m-%dT%H:%M:%SZ")
    power_kw = data['solar']['power'] / 1000  # Convert W to kW
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [timestamp],
        'power_kw': [power_kw]
    })

    return df