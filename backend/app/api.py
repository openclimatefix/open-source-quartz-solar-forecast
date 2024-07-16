from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecast import run_forecast
from datetime import datetime

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ForecastRequestBody(PVSite):
    ts:datetime

@app.post("/forecast/")
async def forecast(request:ForecastRequestBody):
    site=PVSite(latitude=request.latitude, longitude=request.longitude, capacity_kwp=request.capacity_kwp)
    # need to convert timestamp to offset-naive to avoid "can't subtract offset-naive and offset-aware datetimes"
    print(request.ts)
    ts=request.ts.replace(tzinfo=None)
    print(ts)
    df =run_forecast(site,"gb",ts)
    return df.to_dict()