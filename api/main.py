from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from quartz_solar_forecast.pydantic_models import PVSite
from .utils import (
    get_enphase_auth_url,
    get_enphase_access_token,
    get_enphase_data,
    get_solis_data_async,
    get_givenergy_data_sync,
    run_forecast_wrapper_async
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/forecast/no_inverter/")
async def forecast_no_inverter(site: PVSite):
    site.inverter_type = "no_inverter"
    df = await run_forecast_wrapper_async(site)
    return df.to_dict()

@app.get("/enphase/auth_url/")
async def enphase_auth_url():
    return {"auth_url": get_enphase_auth_url()}

@app.post("/enphase/access_token/")
async def enphase_access_token(auth_code: str):
    try:
        access_token = get_enphase_access_token(auth_code)
        return {"access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/forecast/enphase/")
async def forecast_enphase(site: PVSite, access_token: str, system_id: str):
    enphase_data = get_enphase_data(system_id, access_token)
    df = await run_forecast_wrapper_async(site, inverter_data=enphase_data)
    return df.to_dict()

@app.post("/forecast/solis/")
async def forecast_solis(site: PVSite):
    try:
        solis_data = await get_solis_data_async()
        df = await run_forecast_wrapper_async(site, inverter_data=solis_data)
        return df.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Solis data: {str(e)}")

@app.post("/forecast/givenergy/")
async def forecast_givenergy(site: PVSite):
    givenergy_data = get_givenergy_data_sync()
    df = await run_forecast_wrapper_async(site, inverter_data=givenergy_data)
    return df.to_dict()