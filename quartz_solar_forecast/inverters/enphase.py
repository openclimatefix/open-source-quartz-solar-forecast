import http.client
import os
from typing import Optional

import pandas as pd
import json
import base64
from datetime import datetime, timedelta, timezone

from urllib.parse import urlencode

from quartz_solar_forecast.inverters.inverter import AbstractInverter
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnphaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    client_id: str = Field(alias="ENPHASE_CLIENT_ID")
    system_id: str = Field(alias="ENPHASE_SYSTEM_ID")
    api_key: str = Field(alias="ENPHASE_API_KEY")
    client_secret: str = Field(alias="ENPHASE_CLIENT_SECRET")


class EnphaseInverter(AbstractInverter):

    def __init__(self, settings: EnphaseSettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        return get_enphase_data(self.__settings)


def get_enphase_auth_url(settings: Optional[EnphaseSettings] = None):
    """
    Generate the authorization URL for the Enphase API.

    :param settings: the Enphase settings
    :return: Authentication URL
    """
    if settings is None:
        # Because this uses env variables we don't want to set it as a default argument, otherwise it will be evaluated
        # even if the method is not called
        settings = EnphaseSettings()

    client_id = settings.client_id

    redirect_uri = (
        "https://api.enphaseenergy.com/oauth/redirect_uri"  # Or your own redirect URI
    )
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    auth_url = f"https://api.enphaseenergy.com/oauth/authorize?{urlencode(params)}"
    return auth_url


def get_enphase_authorization_code(auth_url):
    """
    Open the authorization URL in a browser and retrieve the authorization code from the redirect URI.

    :param auth_url: Authentication URL to get the code
    :return: The one time code for access to a system
    """
    # Open the authorization URL in a browser
    print(f"Please visit the following URL and authorize the application: {auth_url}")
    print(
        "After authorization, you will be redirected to a URL with the authorization code."
    )
    print("Please copy and paste the full redirect URL here:")
    redirect_url = input()
    # Extract the authorization code from the redirect URL
    code = redirect_url.split("?code=")[1]
    return code


def get_enphase_access_token(auth_code: Optional[str] = None, settings: Optional[EnphaseSettings] = None):
    """
    Obtain an access token for the Enphase API using the Authorization Code Grant flow.
    :param auth_code: Optional authorization code. If not provided, it will be obtained.
    :param settings: Optional Enphase settings
    :return: Access Token
    """
    if settings is None:
        # Because this uses env variables we don't want to set it as a default argument, otherwise it will be evaluated
        # even if the method is not called
        settings = EnphaseSettings()

    client_id = settings.client_id
    client_secret = settings.client_secret

    if auth_code is None:
        auth_url = get_enphase_auth_url(settings)
        auth_code = get_enphase_authorization_code(auth_url)

    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    headers = {
        "Authorization": f"Basic {encoded_credentials}"
    }
    conn.request(
        "POST",
        f"/oauth/token?grant_type=authorization_code&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri&code={auth_code}",
        "",
        headers,
    )
    res = conn.getresponse()
    data = res.read()
    data_json = json.loads(data.decode("utf-8"))
    access_token = data_json["access_token"]
    refresh_token = data_json["refresh_token"]

    # Save tokens to environment variables
    os.environ['ENPHASE_ACCESS_TOKEN'] = access_token
    os.environ['ENPHASE_REFRESH_TOKEN'] = refresh_token

    return access_token


def process_enphase_data(data_json: dict, start_at: int) -> pd.DataFrame:
    # Check if 'intervals' key exists in the response
    if 'intervals' not in data_json:
        return pd.DataFrame(columns=["timestamp", "power_kw"])

    # Initialize an empty list to store the data
    data_list = []

    # Loop through the intervals and collect the data for the last week
    for interval in data_json['intervals']:
        end_at = interval['end_at']
        if end_at >= start_at:
            # Convert to UTC
            timestamp = datetime.fromtimestamp(end_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

            # Append the data to the list
            data_list.append({"timestamp": timestamp, "power_kw": interval['powr'] / 1000})

    # Convert the list to a DataFrame
    live_generation_kw = pd.DataFrame(data_list)

    # If DataFrame is empty, return with correct columns
    if live_generation_kw.empty:
        return pd.DataFrame(columns=["timestamp", "power_kw"])

    # Convert to datetime
    live_generation_kw["timestamp"] = pd.to_datetime(live_generation_kw["timestamp"])

    return live_generation_kw


def get_enphase_data(settings: EnphaseSettings) -> pd.DataFrame:
    """ 
    Get live PV generation data from Enphase API v4
    :param settings: the Enphase settings
    :param enphase_system_id: System ID for Enphase API
    :return: Live PV generation in Watt-hours, assumes to be a floating-point number
    """
    access_token = os.getenv('ENPHASE_ACCESS_TOKEN')

    # If access token is not in environment variables, get a new one
    if not access_token:
        access_token = get_enphase_access_token(settings=settings)

    # Set the start time to 1 week ago
    start_at = int((datetime.now() - timedelta(weeks=1)).timestamp())

    # Set the granularity to week
    granularity = "week"

    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "key": settings.api_key
    }

    # Add the system_id and duration parameters to the URL
    url = f"/api/v4/systems/{settings.system_id}/telemetry/production_micro?start_at={start_at}&granularity={granularity}"
    conn.request("GET", url, headers=headers)

    res = conn.getresponse()
    data = res.read()

    # Decode the data read from the response
    decoded_data = data.decode("utf-8")

    # Convert the decoded data into JSON format
    data_json = json.loads(decoded_data)

    # Process the data using the new function
    live_generation_kw = process_enphase_data(data_json, start_at)

    return live_generation_kw
