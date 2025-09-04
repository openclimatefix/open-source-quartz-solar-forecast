# Solar Forecast API Documentation

## Overview

This API provides solar power forecast data based on the given site information. It has been developed using FastAPI and includes the following key endpoints:

1. `/forecast/`: Generate solar power forecasts.

## Endpoints

### 1. Generate Solar Power Forecast

- **Endpoint:** `/forecast/`
- **Method:** `POST`
- **Description:** This endpoint generates a solar power forecast for a specified site and timestamp. It optionally includes real-time data if available.

#### Request Body:

- **ForecastRequest:**
  - `site` (PVSite, required): The site details for which the forecast is to be generated.
  - `timestamp` (string, optional): The initialization timestamp for the forecast in ISO 8601 format. If not provided, the current time will be used.
  - `live_generation`: (list, optional): A list of generation values, that can be used to help improve the forecast. 

- **PVSite:**
  - `latitude` (float, required): The latitude of the site. Must be between -90 and 90.
  - `longitude` (float, required): The longitude of the site. Must be between -180 and 180.
  - `capacity_kwp` (float, required): The capacity of the site in kilowatts peak (kWp). Must be a positive value.
  - `tilt` (float, optional, default=35): The tilt angle of the solar panels in degrees. Must be between 0 and 90.
  - `orientation` (float, optional, default=180): The orientation angle of the solar panels in degrees, measured from north. Must be between 0 and 360.

#### Response:

- **200 OK**
  - **JSON Structure:**
    ```json
    {
      "timestamp": "2023-08-14 10:00:00",
      "predictions": {
        "power_kw": {"2023-08-14 10:00:00": 3.4,
                     "2023-08-14 10:15:00": 3.5, ... }
      }
    }
    ```
  - `timestamp` (string): The initialization timestamp of the forecast.
  - `predictions` (dictionary): The forecasted power data. 

## Error Handling

All endpoints will return appropriate HTTP status codes. Common responses include:

- **200 OK:** The request was successful.
- **400 Bad Request:** The request was malformed or contained invalid data.
- **500 Internal Server Error:** An unexpected error occurred on the server.

## Example Usage

### Generate Solar Power Forecast

**Request:**

```bash
curl -X POST "http://localhost:8000/forecast/" -H "Content-Type: application/json" -d '{
  "site": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "capacity_kwp": 5.0,
    "tilt": 30,
    "orientation": 180,
  },
  "timestamp": "2023-08-14T10:00:00Z"
}'
```

**Response:**

```json
{
  "timestamp": "2023-08-14 10:00:00",
  "predictions": {
    "power_kw": {"2023-08-14 10:00:00": 3.4,
                 "2023-08-14 10:15:00": 3.5, ... }
  }
}
```


## Development

### Local

Install the repo with
```uv sync```
and then run the fastapi container with
```uv run uvicorn api.v1.api:app --reload ```

### Docker

You can build the docker file using
```docker build . -t open```
And then run it using
```docker run -it -p 8000:8000 open```