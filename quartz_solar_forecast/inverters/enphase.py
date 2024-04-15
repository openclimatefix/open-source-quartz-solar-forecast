import requests
import os

from dotenv import load_dotenv

ENPHASE_CLIENT_ID = os.getenv('ENPHASE_CLIENT_ID')
ENPHASE_CLIENT_SECRET = os.getenv('ENPHASE_CLIENT_SECRET')
ENPHASE_API_KEY = os.getenv('ENPHASE_API_KEY')

def get_enphase_access_token():
    """
    Obtain an access token for the Enphase API using the Client Credentials Grant flow.
    """
    url = "https://api.enphaseenergy.com/oauth/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {(ENPHASE_CLIENT_ID + ':' + ENPHASE_CLIENT_SECRET).encode().decode('utf-8')}",
    }
    data = {
        "grant_type": "client_credentials",
        "scope": "read"
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def get_enphase_data(enphase_system_id: str) -> float:
    """
    Get live PV generation data from Enphase API v4

    :param enphase_system_id: System ID for Enphase API
    :return: Live PV generation in Watt-hours, assumes to be a floating-point number
    """
    url = f'https://api.enphaseenergy.com/api/v4/{enphase_system_id}/summary'
    headers = {
        'Authorization': f'Bearer {get_enphase_access_token()}',
        'key': ENPHASE_API_KEY
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    # Extracting live generation data assuming it's in Watt-hours
    live_generation_wh = data['current_power']['power']
    
    return live_generation_wh