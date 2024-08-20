import requests
import pandas as pd
from datetime import timedelta

from quartz_solar_forecast.inverters.solarman.solarman_model import SolarmanSettings


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
