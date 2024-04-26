import requests
import os

from dotenv import load_dotenv

ENPHASE_CLIENT_ID = os.getenv('ENPHASE_CLIENT_ID')
ENPHASE_CLIENT_SECRET = os.getenv('ENPHASE_CLIENT_SECRET')
ENPHASE_API_KEY = os.getenv('ENPHASE_API_KEY')

import os
from urllib.parse import urlencode

def get_enphase_auth_url():
    """
    Generate the authorization URL for the Enphase API.

    :param None
    :return: Authentication URL
    """
    client_id = os.getenv('ENPHASE_CLIENT_ID')
    redirect_uri = 'https://api.enphaseenergy.com/oauth/redirect_uri' # Or your own redirect URI
    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
    }
    auth_url = f'https://api.enphaseenergy.com/oauth/authorize?{urlencode(params)}'
    return auth_url

def get_enphase_authorization_code(auth_url):
    """
    Open the authorization URL in a browser and retrieve the authorization code from the redirect URI.

    :param auth_url: Authentication URL to get the code
    :return: The one time code for access to a system
    """
    # Open the authorization URL in a browser
    print(f"Please visit the following URL and authorize the application: {auth_url}")
    print("After authorization, you will be redirected to a URL with the authorization code.")
    print("Please copy and paste the full redirect URL here:")
    redirect_url = input()
    # Extract the authorization code from the redirect URL
    code = redirect_url.split('?code=')[1]
    return code

def get_enphase_access_token():
    """
    Obtain an access token for the Enphase API using the Authorization Code Grant flow.
    :param None
    :return: Access Token
    """

    auth_url = get_enphase_auth_url()
    auth_code = get_enphase_authorization_code(auth_url)

    url = "https://api.enphaseenergy.com/oauth/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {(ENPHASE_CLIENT_ID + ':' + ENPHASE_CLIENT_SECRET).encode().decode('utf-8')}",
    }
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': 'https://api.enphaseenergy.com/oauth/redirect_uri', # Or your own redirect URI
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
    auth_url = get_enphase_auth_url()
    auth_code = get_enphase_authorization_code(auth_url)
    access_token = get_enphase_access_token(auth_code)

    url = f'https://api.enphaseenergy.com/api/v4/{enphase_system_id}/summary'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'key': ENPHASE_API_KEY
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Extracting live generation data assuming it's in Watt-hours
    live_generation_wh = data['current_power']
    
    return live_generation_wh