from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecast import run_forecast

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

@app.post("/forecast/")
async def forecast(site:PVSite):
    df =run_forecast(site)
    return df.to_dict()