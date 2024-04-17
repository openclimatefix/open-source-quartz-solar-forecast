import requests
import os

from dotenv import load_dotenv

load_dotenv()

SOLAREDGE_ACCOUNT_KEY = os.getenv('SOLAREDGE_ACCOUNT_KEY')
SOLAREDGE_USER_KEY = os.getenv('SOLAREDGE_USER_KEY')

def get_site_coordinates(site_id: str) -> tuple[float, float]:
    """
    Fetch the latitude and longitude of a SolarEdge site.
    :param site_id: The site ID
    :return: A tuple of (latitude, longitude)
    """
    base_url = "https://monitoringapi.solaredge.com/v2"
    headers = {
        'X-Account-Key': SOLAREDGE_ACCOUNT_KEY,
        'X-API-Key': SOLAREDGE_USER_KEY
    }

    site_details_url = f"{base_url}/sites/{site_id}"
    response = requests.get(site_details_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    latitude = data['location']['latitude']
    longitude = data['location']['longitude']
    return latitude, longitude

def get_site_list():
    """
    Fetch the list of sites associated with the account.
    :return: A list of site IDs
    """
    base_url = "https://monitoringapi.solaredge.com/v2"
    headers = {
        'X-Account-Key': SOLAREDGE_ACCOUNT_KEY,
        'X-API-Key': SOLAREDGE_USER_KEY
    }

    site_list_url = f"{base_url}/sites"
    response = requests.get(site_list_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    site_ids = [site['siteId'] for site in data]
    return site_ids

def get_solaredge_data(site_id: str) -> float:
    """
    Get live PV generation data from the SolarEdge Monitoring API v2.
    :param site_id: Site ID for the SolarEdge API
    :return: Live PV generation in Watt-hours, assumed to be a floating-point number
    """
    base_url = "https://monitoringapi.solaredge.com/v2"
    headers = {
        'X-Account-Key': SOLAREDGE_ACCOUNT_KEY,
        'X-API-Key': SOLAREDGE_USER_KEY
    }

    site_overview_url = f"{base_url}/sites/{site_id}/overview"
    response = requests.get(site_overview_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Extracting live generation data assuming it's in Watt-hours
    live_generation_wh = data['production']['total']
    return live_generation_wh