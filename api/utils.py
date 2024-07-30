import os
import base64
import json
import http.client
import asyncio
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.inverters.enphase import process_enphase_data
from quartz_solar_forecast.inverters.solis import get_solis_data
from quartz_solar_forecast.inverters.givenergy import get_givenergy_data
from quartz_solar_forecast.forecast import run_forecast

load_dotenv()

def get_enphase_auth_url():
    client_id = os.getenv("ENPHASE_CLIENT_ID")
    redirect_uri = "https://api.enphaseenergy.com/oauth/redirect_uri"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    auth_url = f"https://api.enphaseenergy.com/oauth/authorize?{urlencode(params)}"
    return auth_url

def get_enphase_access_token(auth_code):
    client_id = os.getenv("ENPHASE_CLIENT_ID")
    client_secret = os.getenv("ENPHASE_CLIENT_SECRET")

    credentials = f"{client_id}:{client_secret}"
    credentials_bytes = credentials.encode("utf-8")
    encoded_credentials = base64.b64encode(credentials_bytes).decode("utf-8")
    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    payload = ""
    headers = {"Authorization": f"Basic {encoded_credentials}"}
    conn.request(
        "POST",
        f"/oauth/token?grant_type=authorization_code&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri&code={auth_code}",
        payload,
        headers,
    )
    res = conn.getresponse()
    data = res.read()

    decoded_data = data.decode("utf-8")
    data_json = json.loads(decoded_data)

    if "error" in data_json:
        raise ValueError(f"Error in getting access token: {data_json['error_description']}")

    if "access_token" not in data_json:
        raise KeyError(f"Access token not found in response. Response: {data_json}")

    return data_json["access_token"]

def get_enphase_data(enphase_system_id: str, access_token: str) -> pd.DataFrame:
    api_key = os.getenv("ENPHASE_API_KEY")
    start_at = int((datetime.now() - timedelta(weeks=1)).timestamp())
    granularity = "week"

    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    headers = {"Authorization": f"Bearer {str(access_token)}", "key": str(api_key)}

    url = f"/api/v4/systems/{enphase_system_id}/telemetry/production_micro?start_at={start_at}&granularity={granularity}"
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    decoded_data = data.decode("utf-8")
    data_json = json.loads(decoded_data)

    return process_enphase_data(data_json, start_at)

def get_givenergy_data_sync():
    return get_givenergy_data()


def run_forecast_wrapper(site: PVSite, ts=None, inverter_data=None):
    if ts is None:
        ts = pd.Timestamp.now().round("15min")
    
    # For no inverter case, just run the forecast without inverter_data
    if site.inverter_type == "no_inverter":
        return run_forecast(site=site, ts=ts)
    
    return run_forecast(site=site, ts=ts)

async def get_solis_data_async():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_solis_data)

async def run_forecast_wrapper_async(site: PVSite, ts=None, inverter_data=None):
    if ts is None:
        ts = pd.Timestamp.now().round("15min")
    
    if site.inverter_type == "no_inverter":
        return run_forecast(site=site, ts=ts)
    
    # For cases with inverter data, we still need to implement the logic to use this data
    # This is a placeholder and should be updated based on how run_forecast should use inverter_data
    return run_forecast(site=site, ts=ts)
    
    # For cases with inverter data, we still need to implement the logic to use this data
    # This is a placeholder and should be updated based on how run_forecast should use inverter_data
    return run_forecast(site=site, ts=ts)