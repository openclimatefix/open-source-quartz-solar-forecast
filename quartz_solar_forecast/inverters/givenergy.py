import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_givenergy_data():
    """
    Fetch the last 7 days of data from the GivEnergy API and return a DataFrame.
    
    :return: DataFrame with timestamp and power_kw columns
    """
    # Get API details from environment variables
    api_key = os.getenv('GIVENERGY_API_KEY')
    inverter_serial_number = os.getenv('GIVENERGY_INVERTER_SERIAL_NUMBER')
    
    if not api_key or not inverter_serial_number:
        raise ValueError("GIVENERGY_API_KEY or GIVENERGY_INVERTER_SERIAL_NUMBER not set in environment variables")

    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    all_data = []
    
    # Iterate through each day in the past week
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f'https://api.givenergy.cloud/v1/inverter/{inverter_serial_number}/data-points/{date_str}'
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        page = 1
        
        while True:
            params = {'page': str(page)}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}")
            
            data = response.json()['data']
            
            if not data:
                break
            
            all_data.extend(data)
            page += 1
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    # Process the data
    processed_data = []
    for item in all_data:
        timestamp = datetime.strptime(item['time'], "%Y-%m-%dT%H:%M:%SZ")
        if start_date <= timestamp <= end_date:
            power_kw = item['power']['solar']['power'] / 1000  # Convert W to kW
            processed_data.append({
                'timestamp': timestamp,
                'power_kw': power_kw
            })
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    df = df.sort_values('timestamp')
    
    return df