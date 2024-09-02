""" Function to get NWP data and create fake PV dataset"""
import ssl
from datetime import datetime
from typing import Optional

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import xarray as xr
from retry_requests import retry

from quartz_solar_forecast.pydantic_models import PVSite

ssl._create_default_https_context = ssl._create_unverified_context


def get_nwp(site: PVSite, ts: datetime, nwp_source: str = "icon") -> xr.Dataset:
    """
    Get GFS NWP data for a point time space and time

    :param site: the PV site
    :param ts: the timestamp for when you want the forecast for
    :param nwp_source: the nwp data source. Either "gfs", "icon" or "ukmo". Defaults to "icon"
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
        # Getting NWP from open meteo weather forecast API by ICON, GFS, or UKMO within the last 3 months
        if nwp_source == "icon":
            url_nwp_source = "dwd-icon"
            url = f"https://api.open-meteo.com/v1/{url_nwp_source}"
        elif nwp_source == "gfs":
            url_nwp_source = "gfs"
            url = f"https://api.open-meteo.com/v1/{url_nwp_source}"
        elif nwp_source == "ukmo":
            url = "https://api.open-meteo.com/v1/forecast"
        else:
            raise Exception(f'Source ({nwp_source}) must be either "icon", "gfs", or "ukmo"')

    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": f"{start}",
        "end_date": f"{end}",
        "hourly": variables
    }

    # Add the "models" parameter if using "ukmo"
    if nwp_source == "ukmo":
        params["models"] = "ukmo_seamless"

    # Make API call to URL
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
        hourly_data["vis"] = 24000.0

    df = pd.DataFrame(data = hourly_data)
    df = df.set_index("time")
    df = df.astype('float64')

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


def process_pv_data(live_generation_kw: Optional[pd.DataFrame], ts: pd.Timestamp, site: 'PVSite') -> xr.Dataset:
    """
    Process PV data and create an xarray Dataset.

    :param live_generation_kw: DataFrame containing live generation data, or None
    :param ts: Current timestamp
    :param site: PV site information
    :return: xarray Dataset containing processed PV data
    """
    if live_generation_kw is not None and not live_generation_kw.empty:
        # Get the most recent data
        recent_pv_data = live_generation_kw[live_generation_kw['timestamp'] <= ts]
        power_kw = np.array([recent_pv_data["power_kw"].values], dtype=np.float64)
        timestamp = recent_pv_data['timestamp'].values
    else:
        # Make fake PV data; this is where we could add the history of a PV system
        power_kw = np.array([[np.nan]])
        timestamp = [ts]

    da = xr.DataArray(
        data=power_kw,
        dims=["pv_id", "timestamp"],
        coords=dict(
            longitude=(["pv_id"], [site.longitude]),
            latitude=(["pv_id"], [site.latitude]),
            timestamp=timestamp,
            pv_id=[1],
            kwp=(["pv_id"], [site.capacity_kwp]),
            tilt=(["pv_id"], [site.tilt]),
            orientation=(["pv_id"], [site.orientation]),
        ),
    )
    da = da.to_dataset(name="generation_kw")

    return da

def make_pv_data(site: PVSite, ts: pd.Timestamp) -> xr.Dataset:
    """
    Make PV data by combining live data from various inverters.
    
    :param site: the PV site
    :param ts: the timestamp of the site
    :return: The combined PV dataset in xarray form
    """
    live_generation_kw = site.get_inverter().get_data(ts)
    # Process the PV data
    da = process_pv_data(live_generation_kw, ts, site)

    return da
