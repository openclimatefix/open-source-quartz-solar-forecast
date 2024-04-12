""" Function to get NWP data and create fake PV dataset"""
import json
import ssl
from datetime import datetime
import os  # Add import for os module

import numpy as np
import pandas as pd
import requests
import xarray as xr

import openmeteo_requests
import requests_cache
from retry_requests import retry

from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.inverters.enphase import get_enphase_data # Added import for get_enphase_data from /inverters/enphase.py

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
from dotenv import load_dotenv

# Assigning secrets from the .env file
ENPHASE_API_KEY = os.getenv('ENPHASE_API_KEY')
ENPHASE_USER_ID = os.getenv('ENPHASE_USER_ID')

def get_nwp(site: PVSite, ts: datetime, nwp_source: str = "icon") -> xr.Dataset:
    """
    Get GFS NWP data for a point time space and time

    :param site: the PV site
    :param ts: the timestamp for when you want the forecast for
    :param nwp_source: the nwp data source. Either "gfs" or "icon". Defaults to "icon"
    :return: nwp forecast in xarray
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Define the variables we want. Visibility is handled separately after the main request
    variables = [
        "temperature_2m", 
        "precipitation", 
        "cloud_cover_low", 
        "cloud_cover_mid", 
        "cloud_cover_high", 
        "wind_speed_10m", 
        "shortwave_radiation", 
        "direct_radiation"
    ]

    start = ts.date()
    end = start + pd.Timedelta(days=7)

    url = ""

    # check whether the time stamp is more than 3 months in the past
    if (datetime.now() - ts).days > 90:
        print("Warning: The requested timestamp is more than 3 months in the past. The weather data are provided by a reanalyse model and not ICON or GFS.")
        
        # load data from open-meteo Historical Weather API
        url = "https://archive-api.open-meteo.com/v1/archive"

    else:

        # Getting NWP from open meteo weather forecast API by ICON or GFS model within the last 3 months
        url_nwp_source = None
        if nwp_source == "icon":
            url_nwp_source = "dwd-icon"
        elif nwp_source == "gfs":
            url_nwp_source = "gfs"
        else:
            raise Exception(f'Source ({nwp_source}) must be either "icon" or "gfs"')

        url = f"https://api.open-meteo.com/v1/{url_nwp_source}"

    params = {
    	"latitude": site.latitude,
    	"longitude": site.longitude,
    	"start_date": f"{start}",
    	"end_date": f"{end}",
    	"hourly": variables
    }
    response = openmeteo.weather_api(url, params=params)
    hourly = response[0].Hourly()

    hourly_data = {"time": pd.date_range(
    	start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
    	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
    	freq = pd.Timedelta(seconds = hourly.Interval()),
    	inclusive = "left"
    )}

    # variables index as in the variables array of the request
    hourly_data["t"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["prate"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["lcc"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["mcc"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["hcc"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["si10"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["dswrf"] = hourly.Variables(6).ValuesAsNumpy()
    hourly_data["dlwrf"] = hourly.Variables(7).ValuesAsNumpy()

    # handle visibility 
    if (datetime.now() - ts).days <= 90:
        # load data from open-meteo gfs model
        params = {
        	"latitude": site.latitude,
        	"longitude": site.longitude,
        	"start_date": f"{start}",
        	"end_date": f"{end}",
        	"hourly": "visibility"
        }
        data_vis_gfs = openmeteo.weather_api("https://api.open-meteo.com/v1/gfs", params=params)[0].Hourly().Variables(0).ValuesAsNumpy()
        hourly_data["vis"] = data_vis_gfs
    else:
        # set to maximum visibility possible
        hourly_data["vis"] = 24000

    df = pd.DataFrame(data = hourly_data)
    df = df.set_index("time")

    # convert data into xarray
    data_xr = format_nwp_data(df, nwp_source, site)

    return data_xr

def format_nwp_data(df: pd.DataFrame, nwp_source:str, site: PVSite):
    data_xr = xr.DataArray(
        data=df.values,
        dims=["step", "variable"],
        coords=dict(
            step=("step", df.index - df.index[0]),
            variable=df.columns,
        ),
    )
    data_xr = data_xr.to_dataset(name=nwp_source)
    data_xr = data_xr.assign_coords(
        {"x": [site.longitude], "y": [site.latitude], "time": [df.index[0]]}
    )
    return data_xr

def make_pv_data(site: PVSite, ts: pd.Timestamp) -> xr.Dataset:
    """
    Make PV data by combining Enphase live data and fake PV data

    Later we could add PV history here

    :param site: the PV site
    :param ts: the timestamp of the site
    :return: The combined PV dataset in xarray form
    """

    # Check if the site has an inverter and use_enphase_data flag accordingly
    use_enphase_data = site.is_inverter

    if use_enphase_data:
        # Fetch live Enphase data and store it in live_generation_wh
        live_generation_wh = get_enphase_data(ENPHASE_USER_ID, ENPHASE_API_KEY)
    else:
        live_generation_wh = np.nan  # Default value if not using live Enphase data

    # Combine live Enphase data with fake PV data, this is where we could add history of a pv system
    generation_wh = [[live_generation_wh]]
    lon = [site.longitude]
    lat = [site.latitude]
    timestamp = [ts]
    pv_id = [1]

    da = xr.DataArray(
        data=generation_wh,
        dims=["pv_id", "timestamp"],
        coords=dict(
            longitude=(["pv_id"], lon),
            latitude=(["pv_id"], lat),
            timestamp=timestamp,
            pv_id=pv_id,
            kwp=(["pv_id"], [site.capacity_kwp]),
            tilt=(["pv_id"], [site.tilt]),
            orientation=(["pv_id"], [site.orientation]),
        ),
    )
    da = da.to_dataset(name="generation_wh")

    return da