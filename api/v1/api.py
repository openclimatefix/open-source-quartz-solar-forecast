"""Main API."""

from datetime import UTC, datetime
from importlib.metadata import version

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

__version__ = version("quartz_solar_forecast")

description = """
API for [Open Source Quartz Solar Forecast](
    https://github.com/openclimatefix/open-source-quartz-solar-forecast
).

This project aims to provide a free, open-source photovoltaic (PV) forecasting tool
 that is simple to use and works anywhere in the world.

The forecast outputs the expected PV generation (in kW) for up to 48 hours ahead at a single site.

## How it works

- A machine learning model is trained on historical weather and solar generation data from the UK.
- The model has been trained using data from ~25,000 PV sites.
- Forecasts are generated using weather data from open-meteo.com

## Commercial forecasts

Open Climate Fix also provides a commercial PV forecast service.
For more information, please contact: quartz.support@openclimatefix.org

## Example

**Request:**

```bash
curl -X POST "https://open.quartz.solar/forecast/" -H "Content-Type: application/json" -d '{
"site": {
    "latitude": "37.7749",
    "longitude": "-122.4194",
    "capacity_kwp": "5.0",
    "tilt": "30",
    "orientation": "180",
  },
  "timestamp": "2023-08-14T10:00:00Z",
  "live_generation": [
    {
      "timestamp": "2023-08-14T09:45:00Z",
      "generation": 2.5
    },
    {
      "timestamp": "2023-08-14T09:30:00Z",
      "generation": 2.2
    }
  ]
}'
```

Please use UTC timestamps throughout

**Response:**

```json
{
  "timestamp": "2023-08-14 10:00:00",
  "predictions": {
    "power_kw": [values],
  }
}
```
## Want to learn more

We've presented Quartz Solar Forecast at two open source conferences:

- **FOSDEM 2024** (Free and Open source Software Developers' European Meeting):
How we built Open Quartz, our motivation behind it and its impact on aiding organizations
 in resource optimization [Watch the talk](https://www.youtube.com/watch?v=NAZ2VeiN1N8)

- **LF Energy 2024**: Exploring Open Quartz's developments - new models, inverter APIs
 and our Open Source journey at Open Climate Fix
[Watch the talk](https://www.youtube.com/watch?v=YTaq41ztEDg)

And you can always head over to our
[Github page](https://github.com/openclimatefix/open-source-quartz-solar-forecast)

"""


app = FastAPI(description=description, version=__version__, title="Open Quartz Solar Forecast API")

# CORS middleware setup
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastValues(BaseModel):
    power_kw: dict[datetime, float] = Field(
        ..., description="Dictionary mapping timestamps to power predictions in kW."
    )



class GenerationValue(BaseModel):
    """Generation Value"""
    timestamp: datetime
    generation: float



class ForecastResponse(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp for the forecast.")
    predictions: ForecastValues = Field(..., description="Forecasted power values.")


class ForecastRequest(BaseModel):
    site: PVSite = Field(..., description="PV site information.")
    timestamp: datetime | None = Field(
        None, description="Optional timestamp for the forecast request."
    )
    live_generation: list[GenerationValue] | None = Field(
        None, description="Optional list of live generation values."
    )

@app.post("/forecast/")
def forecast(forecast_request: ForecastRequest) -> ForecastResponse:
    """Get a PV Forecast for a site."""
    site = forecast_request.site
    ts = forecast_request.timestamp if forecast_request.timestamp else datetime.now(UTC).isoformat()

    live_generation = forecast_request.live_generation

    timestamp = pd.Timestamp(ts).tz_localize(None)
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # convert live generation to dataframe
    if live_generation is not None and len(live_generation) > 0:
        live_generation_df = pd.DataFrame([g.model_dump() for g in live_generation])
        live_generation_df["timestamp"] = pd.to_datetime(live_generation_df["timestamp"])
        # if timezone, set to UTC and drop it
        if live_generation_df["timestamp"].dt.tz is not None:
            live_generation_df["timestamp"] = live_generation_df["timestamp"].dt.tz_convert("UTC")
            live_generation_df["timestamp"] = live_generation_df["timestamp"].dt.tz_localize(None)

        # make sure timestamps are ordered from earliest to latest
        live_generation_df = live_generation_df.sort_values(by="timestamp")

        live_generation_df.rename(columns={"generation": "power_kw"}, inplace=True)
    else:
        live_generation_df = None



    site_no_live = PVSite(
        latitude=site.latitude,
        longitude=site.longitude,
        capacity_kwp=site.capacity_kwp,
        tilt=site.tilt,
        orientation=site.orientation,
    )

    predictions = run_forecast(site=site_no_live, ts=timestamp, live_generation=live_generation_df)


    response = {
        "timestamp": formatted_timestamp,
        "predictions": predictions.to_dict(),
    }

    return response
