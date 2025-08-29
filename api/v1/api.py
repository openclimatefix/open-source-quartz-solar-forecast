"""Main API."""
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

description = """
API for [Open Source Quartz Solar Forecast](https://github.com/openclimatefix/open-source-quartz-solar-forecast).

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
    "orientation": "180"
  }, 
  "timestamp": "2023-08-14T10:00:00Z"
}'
```

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
 and our Open Source journey at Open Climate Fix [Watch the talk](https://www.youtube.com/watch?v=YTaq41ztEDg)

And you can always head over to our [Github page](https://github.com/openclimatefix/open-source-quartz-solar-forecast)

"""


app = FastAPI(description=description, version="0.0.1", title="Open Quartz Solar Forecast API")

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
    """Dictionary mapping timestamps to power predictions in kW."""
    power_kw: dict[datetime, float]

class ForecastResponse(BaseModel):
    """Response model for forecast predictions."""
    timestamp: datetime
    predictions: ForecastValues

class ForecastRequest(BaseModel):
  site: PVSite
  timestamp: datetime | None = None

@app.post("/forecast/")
def forecast(forecast_request: ForecastRequest) -> ForecastResponse:
    """Get a PV Forecast for a site."""
    site = forecast_request.site
    ts = forecast_request.timestamp if forecast_request.timestamp else datetime.now(UTC).isoformat()

    timestamp = pd.Timestamp(ts).tz_localize(None)
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # TODO add live generation

    site_no_live = PVSite(latitude=site.latitude,
                          longitude=site.longitude,
                          capacity_kwp=site.capacity_kwp, 
                          tilt=site.tilt,
                          orientation=site.orientation)

    predictions = run_forecast(site=site_no_live, ts=timestamp)

    response = {
        "timestamp": formatted_timestamp,
        "predictions": predictions.to_dict(),
    }

    return response
