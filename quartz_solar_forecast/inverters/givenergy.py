import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_givenergy_latest_data():
    """
    Fetch the latest data from the GivEnergy API and return a DataFrame.
    
    :return: DataFrame with timestamp and power_kw columns
    """
    # Get API details from environment variables
    api_key = os.getenv('GIVENERGY_API_KEY')
    inverter_serial_number = os.getenv('GIVENERGY_INVERTER_SERIAL_NUMBER')
    
    if not api_key or not inverter_serial_number:
        raise ValueError("GIVENERGY_API_KEY or GIVENERGY_INVERTER_SERIAL_NUMBER not set in environment variables")

    url = f'https://api.givenergy.cloud/v1/inverter/{inverter_serial_number}/system-data/latest'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()['data']
    
    # Process the data
    timestamp = datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S")
    power_kw = data['solar']['power'] / 1000  # Convert W to kW
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [timestamp],
        'power_kw': [power_kw]
    })
    
    return df