import json
import os
import logging
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite, ForecastRequest, TokenRequest
from quartz_solar_forecast.inverters.enphase import get_enphase_auth_url, get_enphase_access_token, get_enphase_data

load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)

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
def forecast(forecast_request: ForecastRequest):
    site = forecast_request.site
    ts = forecast_request.timestamp if forecast_request.timestamp else datetime.now(timezone.utc).isoformat()

    timestamp = pd.Timestamp(ts).tz_localize(None)
    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    site_no_live = PVSite(latitude=site.latitude, longitude=site.longitude, capacity_kwp=site.capacity_kwp)
    predictions_no_live = run_forecast(site=site_no_live, ts=timestamp)

    if not site.inverter_type:
        predictions = predictions_no_live
    elif site.inverter_type == 'enphase':
        try:
            enphase_system_id = os.getenv('ENPHASE_SYSTEM_ID')
            start_at = int((timestamp - timedelta(weeks=1)).timestamp())
            predictions_with_live = get_enphase_data(enphase_system_id, start_at, "week")
            
            logging.info(f"Predictions with live data:\n{predictions_with_live}")
            logging.info(f"Predictions without live data:\n{predictions_no_live}")

            if predictions_with_live.empty:
                logging.warning("No Enphase data available, using predictions without live data")
                predictions = predictions_no_live.rename(columns={'power_kw': 'power_kw_no_live_pv'})
            else:
                # Ensure both DataFrames have the same index
                common_index = predictions_with_live.index.intersection(predictions_no_live.index)
                predictions_with_live = predictions_with_live.loc[common_index]
                predictions_no_live = predictions_no_live.loc[common_index]
                
                predictions_with_live['power_kw_no_live_pv'] = predictions_no_live['power_kw']
                predictions = predictions_with_live
        except Exception as e:
            logging.error(f"Error fetching Enphase data: {str(e)}")
            predictions = predictions_no_live.rename(columns={'power_kw': 'power_kw_no_live_pv'})
    else:
        predictions_with_live = run_forecast(site=site, ts=timestamp)
        predictions_with_live['power_kw_no_live_pv'] = predictions_no_live['power_kw']
        predictions = predictions_with_live

    # Convert DataFrame to a format that's JSON serializable
    predictions_dict = predictions.reset_index().to_dict(orient='records')

    response = {
        "timestamp": formatted_timestamp,
        "predictions": predictions_dict,
        "raw_enphase_data": predictions_with_live.to_dict(orient='records') if site.inverter_type == 'enphase' else None
    }

    return JSONResponse(content=response)

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
        raise HTTPException(status_code=400, detail=f"Error getting access token and system ID: {str(e)}")