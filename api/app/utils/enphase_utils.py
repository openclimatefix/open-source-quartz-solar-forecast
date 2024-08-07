from quartz_solar_forecast.inverters.enphase import (
    get_enphase_auth_url,
    get_enphase_access_token,
    get_enphase_data
)
from urllib.parse import parse_qs, urlparse

def get_enphase_auth_url_wrapper():
    return get_enphase_auth_url()

def get_enphase_access_token_wrapper(redirect_url):
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)
    auth_code = query_params.get('code', [None])[0]
    if not auth_code:
        raise ValueError("No authorization code found in the redirect URL")
    return get_enphase_access_token(auth_code)

def get_enphase_data_wrapper(system_id, access_token):
    return get_enphase_data(system_id)