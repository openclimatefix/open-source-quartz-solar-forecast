from typing import Optional

import requests
import pandas as pd
from datetime import timedelta, datetime
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from quartz_solar_forecast.inverters.inverter import AbstractInverter


class SolarmanSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

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


def get_solarman_data(start_date, end_date, settings: SolarmanSettings):
    """
    Fetch data from the Solarman API from start_date to end_date.
    
    :param start_date: Start date (datetime object)
    :param end_date: End date (datetime object)
    :param settings: the Solarman settings
    :return: DataFrame with timestamp and power_kw columns
    """
    all_data = []
    
    current_date = start_date
    
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        url = f"{settings.url}/{settings.id}/record"
        
        headers = {
            'Authorization': f'Bearer {settings.token}',
            'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        }
        
        params = {
            'year': year,
            'month': month,
            'day': day
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"API request failed for {current_date} with status code {response.status_code}")
        else:
            data = response.json()
            records = data.get('records', [])
            
            if records:
                df = pd.DataFrame(records)
                df['timestamp'] = pd.to_datetime(df['dateTime'], unit='s')
                all_data.append(df)
        
        current_date += timedelta(days=1)
    
    if not all_data:
        raise ValueError("No data found for the specified date range")
    
    # Concatenate all dataframes
    full_data = pd.concat(all_data, ignore_index=True)
    
    # Select relevant columns
    full_data = full_data[['timestamp', 'generationPower']]

    # Convert watts to kilowatts and rename the column
    full_data['power_kw'] = full_data['generationPower'] / 1000.0
    full_data = full_data.drop('generationPower', axis=1)

    # Ensure the dataframe only has 'timestamp' and 'power_kw' columns
    full_data = full_data[['timestamp', 'power_kw']]

    # Sort by timestamp
    full_data = full_data.sort_values('timestamp')
    
    return full_data
