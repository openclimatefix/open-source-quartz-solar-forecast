import requests

def get_enphase_data(enphase_user_id: str, enphase_api_key: str) -> float:
    """
    Get live PV generation data from Enphase API

    :param enphase_user_id: User ID for Enphase API
    :param enphase_api_key: API Key for Enphase API
    :return: Live PV generation in Watt-hours, assumes to be a floating-point number
    """
    url = f'https://api.enphaseenergy.com/api/v2/systems/{enphase_user_id}/summary'
    headers = {'Authorization': f'Bearer {enphase_api_key}'}

    response = requests.get(url, headers=headers)
    data = response.json()

    # Extracting live generation data assuming it's in Watt-hours
    live_generation_wh = data['current_power']['power']

    return live_generation_wh