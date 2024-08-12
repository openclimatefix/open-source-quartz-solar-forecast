# API Documentation

This document provides an overview of the available endpoints in the FastAPI application for solar energy forecasting.

## Endpoints

### 1. `/forecast/`

- **Method**: `POST`
- **Description**: This endpoint generates a power forecast for a specified solar site. It can optionally use live data from an Enphase system if provided.
- **Request Parameters**: 
  - `site`: A `PVSite` model object containing:
    - `latitude`: (float) The latitude of the solar site.
    - `longitude`: (float) The longitude of the solar site.
    - `capacity_kwp`: (float) The capacity of the solar site in kWp.
    - `inverter_type`: (str) The type of inverter used (e.g., 'enphase').
  - `timestamp` (optional): (str) The timestamp for the forecast in ISO 8601 format. If not provided, the current timestamp in UTC is used.
  - `nwp_source`: (str) The source of numerical weather prediction data.
  - `access_token` (optional): (str) The access token for Enphase data.
  - `enphase_system_id` (optional): (str) The Enphase system ID.
- **Response**:
  - `timestamp`: (str) The timestamp of the forecast in the format `YYYY-MM-DD HH:MM:SS`.
  - `predictions`: (dict) The forecasted power values with keys:
    - `power_kw`: Forecasted power with live data (if available).
    - `power_kw_no_live_pv`: Forecasted power without live data.

### 2. `/solar_inverters/enphase/auth_url`

- **Method**: `GET`
- **Description**: This endpoint provides the authorization URL for Enphase authentication.
- **Response**:
  - `auth_url`: (str) The URL that can be used to initiate the OAuth process with Enphase.

### 3. `/solar_inverters/enphase/token_and_id`

- **Method**: `POST`
- **Description**: This endpoint exchanges an authorization code for an Enphase access token and system ID.
- **Request Parameters**:
  - `redirect_url`: (str) The redirect URL containing the authorization code. Must include `?code=` with the authorization code.
- **Response**:
  - `access_token`: (str) The access token for accessing Enphase data.
  - `enphase_system_id`: (str) The Enphase system ID from the environment variables.
- **Error Responses**:
  - HTTP `400`: If the redirect URL is invalid or if there is an error obtaining the access token and system ID.