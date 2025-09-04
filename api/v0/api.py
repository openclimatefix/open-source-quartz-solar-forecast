import os
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.inverters.enphase import get_enphase_access_token, get_enphase_auth_url
from quartz_solar_forecast.pydantic_models import PVSiteWithInverter, TokenRequest

load_dotenv()

app = FastAPI()

# CORS middleware setup
origins = ["http://localhost:5173", "localhost:5173", "http://localhost:8501", "localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    """
    Request model for generating PV solar forecasts with optional inverter integration.
    
    This model defines the required parameters for requesting a photovoltaic (PV)
    solar power generation forecast for a specific site, with optional real-time
    data integration from supported inverters.
    """
    
    site: PVSiteWithInverter = Field(
        ..., 
        description="PV site details including location, capacity, tilt, orientation, and optional inverter type"
    )
    timestamp: datetime | None = Field(
        default=None, 
        description="Optional timestamp for the forecast. If not provided, current UTC time will be used"
    )


@app.post("/forecast/")
def forecast(forecast_request: ForecastRequest):
    site = forecast_request.site
    ts = forecast_request.timestamp if forecast_request.timestamp else datetime.now(UTC).isoformat()

    timestamp = pd.Timestamp(ts).tz_localize(None)
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    site_no_live = PVSiteWithInverter(
        latitude=site.latitude, longitude=site.longitude, capacity_kwp=site.capacity_kwp
    )
    predictions_no_live = run_forecast(site=site_no_live, ts=timestamp)

    if not site.inverter_type:
        predictions = predictions_no_live
    else:
        predictions_with_live = run_forecast(site=site, ts=timestamp)
        predictions_with_live["power_kw_no_live_pv"] = predictions_no_live["power_kw"]
        predictions = predictions_with_live

    response = {
        "timestamp": formatted_timestamp,
        "predictions": predictions.to_dict(),
    }

    return response


@app.get("/solar_inverters/enphase/auth_url")
def get_enphase_authorization_url():
    auth_url = get_enphase_auth_url()
    return {"auth_url": auth_url}


@app.post("/solar_inverters/enphase/token_and_id")
def get_enphase_token_and_system_id(request: TokenRequest):
    if "?code=" not in request.redirect_url:
        raise HTTPException(status_code=400, detail="Invalid redirect URL")

    auth_code = request.redirect_url.split("?code=")[1]
    try:
        access_token = get_enphase_access_token(auth_code)
        enphase_system_id = os.getenv("ENPHASE_SYSTEM_ID")
        return {"access_token": access_token, "enphase_system_id": enphase_system_id}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error getting access token and system ID: {str(e)}"
        ) from e
