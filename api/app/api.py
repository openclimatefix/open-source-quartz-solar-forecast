import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging
from dotenv import load_dotenv

from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.inverters.enphase import (
    get_enphase_auth_url,
    get_enphase_access_token,
    get_enphase_authorization_code,
)
from quartz_solar_forecast.inverters.solis import get_solis_data
from quartz_solar_forecast.inverters.givenergy import get_givenergy_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware setup
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

@app.post("/forecast/")
async def forecast(request: ForecastRequest):
    site = request.site
    ts = request.timestamp if request.timestamp else pd.Timestamp.now(tz='UTC').isoformat()
    nwp_source = request.nwp_source
    access_token = request.access_token
    enphase_system_id = request.enphase_system_id

    try:
        logging.info(f"Received forecast request: {request}")

        timestamp = pd.Timestamp(ts).tz_localize(None)
        formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Processed forecast request: site={site} timestamp={formatted_timestamp} nwp_source={nwp_source} enphase_system_id={enphase_system_id}")

        solis_data = None
        givenergy_data = None
        
        if site.inverter_type:
            if site.inverter_type == "solis":
                solis_data = await get_solis_data()
            elif site.inverter_type == "givenergy":
                givenergy_data = get_givenergy_data()

            # Run forecast with inverter data
            predictions_with_recent_pv_df = run_forecast(
                site=site,
                ts=timestamp,
                nwp_source=nwp_source,
                access_token=access_token,
                enphase_system_id=enphase_system_id,
                solis_data=solis_data,
                givenergy_data=givenergy_data
            )

            # Run forecast without inverter data
            site_no_live = PVSite(latitude=site.latitude, longitude=site.longitude, capacity_kwp=site.capacity_kwp)
            predictions_df = run_forecast(site=site_no_live, ts=timestamp, nwp_source=nwp_source)

            predictions_with_recent_pv_df["power_kw_no_live_pv"] = predictions_df["power_kw"]
            final_predictions = predictions_with_recent_pv_df
        else:
            # Run forecast without inverter data
            final_predictions = run_forecast(
                site=site,
                ts=timestamp,
                nwp_source=nwp_source
            )

        logging.info(f"Generated predictions: {final_predictions}")

        response = {
            "timestamp": formatted_timestamp,
            "predictions": final_predictions.to_dict()
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
        # Extract the authorization code from the full URL
        auth_code = request.full_auth_url.split("?code=")[1]
        
        # Set the authorization code as an environment variable
        os.environ['ENPHASE_AUTH_CODE'] = auth_code
        
        # Get the auth URL (not used, but required for the function call)
        auth_url = get_enphase_auth_url()
        
        # Call get_enphase_authorization_code (it will use the environment variable)
        authorization_code = get_enphase_authorization_code(auth_url)
        
        # Call the get_enphase_access_token function
        access_token = get_enphase_access_token()
        
        # Remove the environment variable after use
        os.environ.pop('ENPHASE_AUTH_CODE', None)
        
        return {"access_token": access_token}
    except Exception as e:
        logging.error(f"Error in get_enphase_token: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))