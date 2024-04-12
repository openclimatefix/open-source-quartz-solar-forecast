import requests

def get_enphase_data(enphase_system_id: str, enphase_api_key: str, enphase_access_token: str) -> float:
    """
    Get live PV generation data from Enphase API v4

    :param enphase_system_id: System ID for Enphase API
    :param enphase_api_key: API Key for Enphase API
    :param enphase_access_token: Access token for Enphase API
    :return: Live PV generation in Watt-hours, assumes to be a floating-point number
    """
    url = f'https://api.enphaseenergy.com/api/v4/{enphase_system_id}/summary'
    headers = {
        'Authorization': f'Bearer {enphase_access_token}',
        'key': enphase_api_key
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    # Extracting live generation data assuming it's in Watt-hours
    live_generation_wh = data['current_power']['power']
    
    return live_generation_wh