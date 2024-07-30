import base64
from datetime import datetime, timedelta, timezone
import http
import json
from urllib.parse import parse_qs, urlparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import xarray as xr
import logging
import os
from dotenv import load_dotenv

from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecast import predict_tryolabs
from quartz_solar_forecast.data import get_nwp, process_pv_data
from quartz_solar_forecast.forecasts import forecast_v1_tilt_orientation
from quartz_solar_forecast.inverters.enphase import process_enphase_data, get_enphase_auth_url
from quartz_solar_forecast.inverters.solis import get_solis_data
from quartz_solar_forecast.inverters.givenergy import get_givenergy_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost:8501",
    "localhost:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ForecastRequest(BaseModel):
    site: PVSite
    timestamp: Optional[str] = None
    nwp_source: Optional[str] = "icon"
    access_token: Optional[str] = None
    enphase_system_id: Optional[str] = None

class AuthUrlRequest(BaseModel):
    full_auth_url: str

def get_enphase_data(enphase_system_id: str, access_token: str) -> pd.DataFrame:
    """ 
    Get live PV generation data from Enphase API v4
    :param enphase_system_id: System ID for Enphase API
    :param access_token: Access token for Enphase API
    :return: Live PV generation in Watt-hours, assumes to be a floating-point number
    """
    api_key = os.getenv('ENPHASE_API_KEY')

    # Set the start time to 1 week from now
    start_at = int((datetime.now() - timedelta(weeks=1)).timestamp())

    # Set the granularity to week
    granularity = "week"

    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "key": str(api_key)
    }

    # Add the system_id and duration parameters to the URL
    url = f"/api/v4/systems/{enphase_system_id}/telemetry/production_micro?start_at={start_at}&granularity={granularity}"
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

def make_pv_data(
    site: PVSite,
    ts: pd.Timestamp,
    access_token: Optional[str] = None,
    enphase_system_id: Optional[str] = None,
    solis_data: Optional[pd.DataFrame] = None,
    givenergy_data: Optional[pd.DataFrame] = None
) -> xr.Dataset:
    live_generation_kw = None

    if site.inverter_type == "enphase" and enphase_system_id and access_token:
        live_generation_kw = get_enphase_data(enphase_system_id, access_token)
    elif site.inverter_type == "solis" and solis_data is not None:
        live_generation_kw = solis_data
    elif site.inverter_type == "givenergy" and givenergy_data is not None:
        live_generation_kw = givenergy_data

    da = process_pv_data(live_generation_kw, ts, site)
    return da


def predict_ocf(
    site: PVSite,
    model=None,
    ts: datetime = None,
    nwp_source: str = "icon",
    access_token: str = None,
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None
):
    if ts is None:
        ts = pd.Timestamp.now().round("15min")

    nwp_xr = get_nwp(site=site, ts=ts, nwp_source=nwp_source)
    pv_xr = make_pv_data(
        site=site, ts=ts, access_token=access_token, enphase_system_id=enphase_system_id, 
        solis_data=solis_data, givenergy_data=givenergy_data
    )

    pred_df = forecast_v1_tilt_orientation(nwp_source, nwp_xr, pv_xr, ts, model=model)
    return pred_df

def run_forecast_api(
    site: PVSite,
    model: str = "gb",
    ts: datetime = None,
    nwp_source: str = "icon",
    access_token: str = None,  # Add access_token parameter
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None
) -> pd.DataFrame:
    if model == "gb":
        pred_df = predict_ocf(
            site=site,
            model=None,
            ts=ts,
            nwp_source=nwp_source,
            access_token=access_token,  # Pass access_token to predict_ocf
            enphase_system_id=enphase_system_id,
            solis_data=solis_data,
            givenergy_data=givenergy_data
        )
    elif model == "xgb":
        pred_df = predict_tryolabs(site, ts)
    else:
        raise ValueError(f"Unsupported model: {model}. Choose between 'xgb' and 'gb'")

    # Always calculate the no_live_pv forecast
    site_no_inverter = PVSite(
        latitude=site.latitude,
        longitude=site.longitude,
        capacity_kwp=site.capacity_kwp,
        inverter_type=""
    )
    pred_df_no_live_pv = predict_ocf(
        site=site_no_inverter,
        model=None,
        ts=ts,
        nwp_source=nwp_source,
        access_token=access_token
    )

    # Combine the results
    pred_df['power_kw_no_live_pv'] = pred_df_no_live_pv['power_kw']

    return pred_df


def get_enphase_access_token(auth_code):
    """
    Obtain an access token for the Enphase API using the Authorization Code Grant flow.
    :param auth_code: The authorization code received from the redirect
    :return: Access Token
    """
    try:
        client_id = os.getenv('ENPHASE_CLIENT_ID')
        client_secret = os.getenv('ENPHASE_CLIENT_SECRET')

        if not client_id or not client_secret:
            raise ValueError("ENPHASE_CLIENT_ID or ENPHASE_CLIENT_SECRET not set in environment variables")

        credentials = f"{client_id}:{client_secret}"
        credentials_bytes = credentials.encode("utf-8")
        encoded_credentials = base64.b64encode(credentials_bytes)
        encoded_credentials_str = encoded_credentials.decode("utf-8")

        conn = http.client.HTTPSConnection("api.enphaseenergy.com")
        payload = ""
        headers = {
            "Authorization": f"Basic {encoded_credentials_str}"
        }
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

        if 'access_token' not in data_json:
            logging.error(f"Unexpected response from Enphase API: {data_json}")
            raise ValueError("Access token not found in Enphase API response")

        access_token = data_json["access_token"]
        return access_token
    except Exception as e:
        logging.error(f"Error in get_enphase_access_token: {str(e)}")
        raise

@app.post("/forecast/")
async def forecast(request: ForecastRequest):
    site = request.site
    ts = request.timestamp if request.timestamp else pd.Timestamp.now(tz='UTC').isoformat()
    nwp_source = request.nwp_source
    access_token = request.access_token
    enphase_system_id = request.enphase_system_id

    try:
        logging.info(f"Received forecast request: {request}")

        # Convert timestamp string to datetime object
        timestamp = pd.Timestamp(ts).tz_localize(None)  # Remove timezone if present
        
        # Convert timestamp to desired format using the specified method
        end_at = timestamp.timestamp()
        formatted_timestamp = datetime.fromtimestamp(end_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Processed forecast request: site={site} timestamp={formatted_timestamp} nwp_source={nwp_source} enphase_system_id={enphase_system_id}")

        solis_data = None
        givenergy_data = None
        
        if site.inverter_type == "solis":
            solis_data = await get_solis_data()
        elif site.inverter_type == "givenergy":
            givenergy_data = get_givenergy_data()

        predictions_df = run_forecast_api(
            site=site,
            ts=timestamp,
            nwp_source=nwp_source,
            access_token=access_token,
            enphase_system_id=enphase_system_id,
            solis_data=solis_data,
            givenergy_data=givenergy_data
        )

        # Add 'power_kw_no_live_pv' column if it doesn't exist
        if 'power_kw_no_live_pv' not in predictions_df.columns:
            predictions_df['power_kw_no_live_pv'] = None

        # Convert timestamps in DataFrame to desired format
        if 'power_kw' in predictions_df.columns:
            predictions_df.index = pd.to_datetime(predictions_df.index).strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Generated predictions: {predictions_df}")

        response = {
            "timestamp": formatted_timestamp,
            "predictions": predictions_df.to_dict()
        }
        
        return response
    except Exception as e:
        logging.error(f"Error in forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/enphase/auth_url")
async def get_enphase_authorization_url():
    auth_url = get_enphase_auth_url()
    return {"auth_url": auth_url}

@app.post("/enphase/access_token")
async def get_enphase_token(request: AuthUrlRequest):
    try:
        parsed_url = urlparse(request.full_auth_url)
        query_params = parse_qs(parsed_url.query)
        auth_code = query_params.get('code', [None])[0]
        
        if not auth_code:
            raise ValueError("Authorization code not found in the provided URL")

        access_token = get_enphase_access_token(auth_code)
        return {"access_token": access_token}
    except Exception as e:
        logging.error(f"Error in get_enphase_token: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))