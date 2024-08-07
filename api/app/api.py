import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from .schemas import ForecastRequest, AuthUrlRequest
from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.inverters.enphase import (
    get_enphase_auth_url,
    get_enphase_access_token,
    set_enphase_auth_code,
)

# Load environment variables
load_dotenv()

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

@app.post("/forecast/")
def forecast(request: ForecastRequest):
    site = request.site
    ts = request.timestamp if request.timestamp else pd.Timestamp.now(tz='UTC').isoformat()
    nwp_source = request.nwp_source
    access_token = request.access_token
    enphase_system_id = request.enphase_system_id

    timestamp = pd.Timestamp(ts).tz_localize(None)
    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    if site.inverter_type:
        if site.inverter_type == "enphase" and access_token and enphase_system_id:
            # Set the Enphase auth code if provided
            set_enphase_auth_code(access_token)

        # Run forecast with inverter data
        predictions_with_recent_pv_df = run_forecast(
            site=site,
            ts=timestamp,
            nwp_source=nwp_source
        )

        # Run forecast without inverter data
        site_no_live = PVSite(latitude=site.latitude, longitude=site.longitude, capacity_kwp=site.capacity_kwp)
        predictions_df = run_forecast(site=site_no_live, ts=timestamp, nwp_source=nwp_source)

        predictions_with_recent_pv_df["power_kw_no_live_pv"] = predictions_df["power_kw"]
        final_predictions = predictions_with_recent_pv_df

        # Clear the Enphase auth code after use
        if site.inverter_type == "enphase":
            os.environ.pop('ENPHASE_AUTH_CODE', None)
    else:
        # Run forecast without inverter data
        final_predictions = run_forecast(
            site=site,
            ts=timestamp,
            nwp_source=nwp_source
        )

    response = {
        "timestamp": formatted_timestamp,
        "predictions": final_predictions.to_dict()
    }
    
    return response

@app.get("/enphase/auth_url")
def get_enphase_authorization_url():
    auth_url = get_enphase_auth_url()
    return {"auth_url": auth_url}

@app.post("/enphase/access_token")
def get_enphase_token(request: AuthUrlRequest):
    try:
        # Extract the authorization code from the full URL
        auth_code = request.full_auth_url.split("?code=")[1]
        
        # Set the authorization code using the new function
        set_enphase_auth_code(auth_code)
        
        # Call get_enphase_access_token function without arguments
        access_token = get_enphase_access_token()
        
        # Remove the environment variable after use
        os.environ.pop('ENPHASE_AUTH_CODE', None)
        
        return {"access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))