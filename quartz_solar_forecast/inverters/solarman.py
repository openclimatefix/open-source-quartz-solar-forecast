import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SOLARMAN_API_URL = os.getenv('SOLARMAN_API_URL')
SOLARMAN_TOKEN = os.getenv('SOLARMAN_TOKEN')
SOLARMAN_ID = os.getenv('SOLARMAN_ID')

def get_device_info():
    """
    Fetch the device information from the Solarman API.
    
    :return: Dictionary containing deviceId and deviceSn
    """
    url = f"{SOLARMAN_API_URL}/device/v1.0/list"
    
    headers = {
        'Authorization': f'Bearer {SOLARMAN_TOKEN}',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    }
    
    payload = {
        "page": 1,
        "size": 10
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Device list API request failed with status code {response.status_code}")
    
    data = response.json()
    print("DATA: ", data)
    device_list = data.get('deviceList', [])
    
    if not device_list:
        raise ValueError("No devices found")
    
    device = next((dev for dev in device_list if dev['deviceSn'] == SOLARMAN_ID), None)
    
    if not device:
        raise ValueError(f"Device with ID {SOLARMAN_ID} not found")
    
    return {
        'deviceId': device['deviceId'],
        'deviceSn': device['deviceSn']
    }

def get_solarman_data():
    """
    Fetch data from the Solarman API for the past week and return a DataFrame.
    
    :return: DataFrame with timestamp and energy_kwh columns
    """
    device_info = get_device_info()
    
    url = f"{SOLARMAN_API_URL}/historical"
    
    headers = {
        'Authorization': f'Bearer {SOLARMAN_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Get data for the last week
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)
    
    payload = {
        "timeType": 2,  # Daily statistics
        "startTime": start_time.strftime("%Y-%m-%d"),
        "endTime": end_time.strftime("%Y-%m-%d"),
        "deviceSn": device_info['deviceSn'],
        "deviceId": device_info['deviceId']
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Historical data API request failed with status code {response.status_code}")
    
    data = response.json()
    param_data_list = data.get('paramDataList', [])
    
    if not param_data_list:
        raise ValueError("No data found for the specified time range")
    
    # Process the data
    rows = []
    for entry in param_data_list:
        timestamp = datetime.strptime(entry['collectTime'], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        generation_data = next((item for item in entry['dataList'] if item['key'] == 'generation'), None)
        if generation_data:
            energy_kwh = float(generation_data['value'])
            rows.append({'timestamp': timestamp, 'energy_kwh': energy_kwh})
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp')
    
    print(df)
    return df


if __name__ == "__main__":
    get_solarman_data()