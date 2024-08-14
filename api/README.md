# Solar Forecast API Documentation

## Overview

This API provides solar power forecast data based on the given site information and handles authorization with Enphase solar inverters. It has been developed using FastAPI and includes the following key endpoints:

1. `/forecast/`: Generate solar power forecasts.
2. `/solar_inverters/enphase/auth_url`: Retrieve the Enphase authorization URL.
3. `/solar_inverters/enphase/token_and_id`: Obtain an Enphase access token and system ID.

## Endpoints

### 1. Generate Solar Power Forecast

- **Endpoint:** `/forecast/`
- **Method:** `POST`
- **Description:** This endpoint generates a solar power forecast for a specified site and timestamp. It optionally includes real-time data from inverters if available.

#### Request Body:

- **ForecastRequest:**
  - `site` (PVSite, required): The site details for which the forecast is to be generated.
  - `timestamp` (string, optional): The timestamp for the forecast in ISO 8601 format. If not provided, the current time will be used.

- **PVSite:**
  - `latitude` (float, required): The latitude of the site. Must be between -90 and 90.
  - `longitude` (float, required): The longitude of the site. Must be between -180 and 180.
  - `capacity_kwp` (float, required): The capacity of the site in kilowatts peak (kWp). Must be a positive value.
  - `tilt` (float, optional, default=35): The tilt angle of the solar panels in degrees. Must be between 0 and 90.
  - `orientation` (float, optional, default=180): The orientation angle of the solar panels in degrees, measured from north. Must be between 0 and 360.
  - `inverter_type` (string, optional): The type of inverter used. Accepted values: `"enphase"`, `"solis"`, `"givenergy"`, `"solarman"`, or `None`.

#### Response:

- **200 OK**
  - **JSON Structure:**
    ```json
    {
      "timestamp": "2023-08-14 10:00:00",
      "predictions": {
        "power_kw": [values],
        "power_kw_no_live_pv": [values]
      }
    }
    ```
  - `timestamp` (string): The formatted timestamp of the forecast.
  - `predictions` (dictionary): The forecasted power data. If inverter data is available, it will also include `power_kw_no_live_pv` without inverter data.

### 2. Retrieve Enphase Authorization URL

- **Endpoint:** `/solar_inverters/enphase/auth_url`
- **Method:** `GET`
- **Description:** This endpoint returns the authorization URL required to initiate the OAuth flow for Enphase inverters.

#### Response:

- **200 OK**
  - **JSON Structure:**
    ```json
    {
      "auth_url": "https://api.enphaseenergy.com/oauth/authorize?client_id=..."
    }
    ```
  - `auth_url` (string): The URL to redirect the user to for Enphase authorization.

### 3. Obtain Enphase Access Token and System ID

- **Endpoint:** `/solar_inverters/enphase/token_and_id`
- **Method:** `POST`
- **Description:** This endpoint exchanges an authorization code for an access token and retrieves the system ID of the Enphase solar inverter.

#### Request Body:

- **TokenRequest:**
  - `redirect_url` (string, required): The URL to which the user was redirected after Enphase authorization, containing the authorization code.

#### Response:

- **200 OK**
  - **JSON Structure:**
    ```json
    {
      "access_token": "abc123...",
      "enphase_system_id": "123456789"
    }
    ```
  - `access_token` (string): The access token for Enphase API.
  - `enphase_system_id` (string): The system ID of the Enphase solar inverter.

- **400 Bad Request**
  - **Error Message:**
    ```json
    {
      "detail": "Invalid redirect URL"
    }
    ```
  - **Description:** The request was not properly formatted or did not contain the necessary authorization code.

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
    "inverter_type": "enphase"
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
    "power_kw_no_live_pv": [values]
  }
}
```

### Retrieve Enphase Authorization URL

**Request:**

```bash
curl -X GET "http://localhost:8000/solar_inverters/enphase/auth_url"
```

**Response:**

```json
{
  "auth_url": "https://api.enphaseenergy.com/oauth/authorize?client_id=..."
}
```

### Obtain Enphase Access Token and System ID

**Request:**

```bash
curl -X POST "http://localhost:8000/solar_inverters/enphase/token_and_id" -H "Content-Type: application/json" -d '{
  "redirect_url": "https://yourapp.com/callback?code=abc123"
}'
```

**Response:**

```json
{
  "access_token": "abc123...",
  "enphase_system_id": "123456789"
}
```