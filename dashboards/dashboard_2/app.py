import streamlit as st
import http.client
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
import sys
import os
import logging
import xarray as xr
from dotenv import load_dotenv
import base64
import json
from urllib.parse import urlencode
from PIL import Image
import asyncio

from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecasts import forecast_v1_tilt_orientation
from quartz_solar_forecast.forecast import predict_tryolabs
from quartz_solar_forecast.data import get_nwp, process_pv_data
from quartz_solar_forecast.inverters.enphase import process_enphase_data
from quartz_solar_forecast.inverters.solis import get_solis_data
from quartz_solar_forecast.inverters.givenergy import get_givenergy_data
from quartz_solar_forecast.inverters.solarman import get_solarman_data 

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to logo.png
logo_path = os.path.join(script_dir, "logo.png")
im = Image.open(logo_path)

st.set_page_config(
    page_title="Open Source Quartz Solar Forecast | Open Climate Fix",
    layout="wide",
    page_icon=im,
)
st.title("☀️ Open Source Quartz Solar Forecast")

def get_enphase_auth_url():
    client_id = os.getenv("ENPHASE_CLIENT_ID")
    redirect_uri = "https://api.enphaseenergy.com/oauth/redirect_uri"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    auth_url = f"https://api.enphaseenergy.com/oauth/authorize?{urlencode(params)}"
    return auth_url

def get_enphase_access_token(auth_code):
    client_id = os.getenv("ENPHASE_CLIENT_ID")
    client_secret = os.getenv("ENPHASE_CLIENT_SECRET")

    credentials = f"{client_id}:{client_secret}"
    credentials_bytes = credentials.encode("utf-8")
    encoded_credentials = base64.b64encode(credentials_bytes).decode("utf-8")
    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    payload = ""
    headers = {"Authorization": f"Basic {encoded_credentials}"}
    conn.request(
        "POST",
        f"/oauth/token?grant_type=authorization_code&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri&code={auth_code}",
        payload,
        headers,
    )
    res = conn.getresponse()
    data = res.read()

    # Decode the data read from the response
    decoded_data = data.decode("utf-8")

    # Convert the decoded data into JSON format
    data_json = json.loads(decoded_data)

    if "error" in data_json:
        raise ValueError(
            f"Error in getting access token: {data_json['error_description']}"
        )

    if "access_token" not in data_json:
        raise KeyError(f"Access token not found in response. Response: {data_json}")

    access_token = data_json["access_token"]
    return access_token

def get_enphase_data(enphase_system_id: str, access_token: str) -> pd.DataFrame:
    api_key = os.getenv("ENPHASE_API_KEY")
    start_at = int((datetime.now() - timedelta(weeks=1)).timestamp())
    granularity = "week"

    conn = http.client.HTTPSConnection("api.enphaseenergy.com")
    headers = {"Authorization": f"Bearer {str(access_token)}", "key": str(api_key)}

    url = f"/api/v4/systems/{enphase_system_id}/telemetry/production_micro?start_at={start_at}&granularity={granularity}"
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    decoded_data = data.decode("utf-8")
    data_json = json.loads(decoded_data)

    return process_enphase_data(data_json, start_at)

def enphase_authorization():
    if "access_token" not in st.session_state:
        auth_url = get_enphase_auth_url()
        st.write("Please visit the following URL to authorize the application:")
        st.markdown(f"[Enphase Authorization URL]({auth_url})")
        st.write(
            "After authorization, you will be redirected to a URL. Please copy the entire URL and paste it below:"
        )

        redirect_url = st.text_input("Enter the redirect URL:")

        if redirect_url:
            if "?code=" not in redirect_url:
                st.error(
                    "Invalid redirect URL. Please make sure you copied the entire URL."
                )
                return None, None

            auth_code = redirect_url.split("?code=")[1]

            try:
                access_token = get_enphase_access_token(auth_code)
                st.session_state["access_token"] = access_token
                return access_token, os.getenv("ENPHASE_SYSTEM_ID")
            except Exception as e:
                st.error(f"Error getting access token: {str(e)}")
                return None, None
    else:
        return st.session_state["access_token"], os.getenv("ENPHASE_SYSTEM_ID")

    return None, None

def make_pv_data(
    site: PVSite,
    ts: pd.Timestamp,
    access_token: str = None,
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None,
    solarman_data: pd.DataFrame = None
) -> xr.Dataset:
    live_generation_kw = None

    if site.inverter_type == "enphase" and access_token and enphase_system_id:
        live_generation_kw = get_enphase_data(enphase_system_id, access_token)
    elif site.inverter_type == "solis" and solis_data is not None:
        live_generation_kw = solis_data
    elif site.inverter_type == "givenergy" and givenergy_data is not None:
        live_generation_kw = givenergy_data
    elif site.inverter_type == "solarman" and solarman_data is not None:
        live_generation_kw = solarman_data

    da = process_pv_data(live_generation_kw, ts, site)
    return da

def predict_ocf(
    site: PVSite,
    model=None,
    ts: datetime | str = None,
    nwp_source: str = "icon",
    access_token: str = None,
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None,
    solarman_data: pd.DataFrame = None
):
    if ts is None:
        ts = pd.Timestamp.now().round("15min")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    nwp_xr = get_nwp(site=site, ts=ts, nwp_source=nwp_source)
    pv_xr = make_pv_data(
        site=site, ts=ts, access_token=access_token, enphase_system_id=enphase_system_id, 
        solis_data=solis_data, givenergy_data=givenergy_data, solarman_data=solarman_data
    )

    pred_df = forecast_v1_tilt_orientation(nwp_source, nwp_xr, pv_xr, ts, model=model)
    return pred_df

def run_forecast(
    site: PVSite,
    model: str = "gb",
    ts: datetime | str = None,
    nwp_source: str = "icon",
    access_token: str = None,
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None,
    solarman_data: pd.DataFrame = None
) -> pd.DataFrame:
    if model == "gb":
        return predict_ocf(site, None, ts, nwp_source, access_token, enphase_system_id, solis_data, givenergy_data, solarman_data)
    elif model == "xgb":
        return predict_tryolabs(site, ts)
    else:
        raise ValueError(f"Unsupported model: {model}. Choose between 'xgb' and 'gb'")

def fetch_data_and_run_forecast(
    site: PVSite,
    access_token: str = None,
    enphase_system_id: str = None,
    solis_data: pd.DataFrame = None,
    givenergy_data: pd.DataFrame = None,
    solarman_data: pd.DataFrame = None
):
    with st.spinner("Running forecast..."):
        try:
            timestamp = datetime.now().timestamp()
            timestamp_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            ts = pd.to_datetime(timestamp_str)

            # Run forecast with the given site
            predictions_with_inverter = run_forecast(
                site=site,
                ts=ts,
                access_token=access_token,
                enphase_system_id=enphase_system_id,
                solis_data=solis_data,
                givenergy_data=givenergy_data,
                solarman_data=solarman_data
            )

            # Create a site without inverter for comparison
            site_without_inverter = PVSite(
                latitude=site.latitude,
                longitude=site.longitude,
                capacity_kwp=site.capacity_kwp
            )
            predictions_without_inverter = run_forecast(site=site_without_inverter, ts=ts)

            # Combine the results
            predictions_df = predictions_with_inverter.copy()
            predictions_df["power_kw_no_live_pv"] = predictions_without_inverter["power_kw"]

            return predictions_df, ts

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None, None

# Main app logic
st.sidebar.header("PV Site Configuration")

use_defaults = st.sidebar.checkbox("Use Default Values", value=True)

if use_defaults:
    latitude = 51.75
    longitude = -1.25
    capacity_kwp = 1.25
    st.sidebar.text(f"Default Latitude: {latitude}")
    st.sidebar.text(f"Default Longitude: {longitude}")
    st.sidebar.text(f"Default Capacity (kWp): {capacity_kwp}")
else:
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=51.75, step=0.01)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-1.25, step=0.01)
    capacity_kwp = st.sidebar.number_input("Capacity (kWp)", min_value=0.1, value=1.25, step=0.01)

inverter_type = st.sidebar.selectbox("Select Inverter", ["No Inverter", "Enphase", "Solis", "GivEnergy", "Solarman"])

access_token = None
enphase_system_id = None
solis_data = None
givenergy_data = None
solarman_data = None

if inverter_type == "Enphase":
    if "access_token" not in st.session_state:
        access_token, enphase_system_id = enphase_authorization()
    else:
        access_token, enphase_system_id = st.session_state["access_token"], os.getenv(
            "ENPHASE_SYSTEM_ID"
        )

if st.sidebar.button("Run Forecast"):
    if inverter_type == "Enphase" and (access_token is None or enphase_system_id is None):
        st.error(
            "Enphase authorization is required. Please complete the authorization process."
        )
    else:
        # Create PVSite object with user-input or default values
        site = PVSite(
            latitude=latitude,
            longitude=longitude,
            capacity_kwp=capacity_kwp,
            inverter_type=inverter_type.lower()
        )
        
        # Define start_date and end_date for Solarman data
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Fetch data based on the selected inverter type
        if inverter_type == "Enphase":
            predictions_df, ts = fetch_data_and_run_forecast(
                site, access_token, enphase_system_id
            )
        elif inverter_type == "Solis":
            solis_df = asyncio.run(get_solis_data())
            predictions_df, ts = fetch_data_and_run_forecast(
                site, solis_data=solis_df
            )
        elif inverter_type == "GivEnergy":
            givenergy_df = get_givenergy_data()
            predictions_df, ts = fetch_data_and_run_forecast(
                site, givenergy_data=givenergy_df
            )
        elif inverter_type == "Solarman":
            solarman_df = get_solarman_data(start_date=start_date, end_date=end_date)
            predictions_df, ts = fetch_data_and_run_forecast(
                site, solarman_data=solarman_df
            )
        else:
            predictions_df, ts = fetch_data_and_run_forecast(site)

        if predictions_df is not None:
            st.success("Forecast completed successfully!")

            # Display current timestamp
            st.subheader(f"Forecast generated at: {ts}")

            # Create three columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Power", f"{predictions_df['power_kw'].iloc[-1]:.2f} kW"
                )

            with col2:
                total_energy = (
                    predictions_df["power_kw"].sum() * 0.25
                )  # Assuming 15-minute intervals
                st.metric("Total Forecasted Energy", f"{total_energy:.2f} kWh")

            with col3:
                peak_power = predictions_df["power_kw"].max()
                st.metric("Peak Forecasted Power", f"{peak_power:.2f} kW")

            # Create a line chart of power generation
            fig = px.line(
                predictions_df.reset_index(),
                x="index",
                y=["power_kw", "power_kw_no_live_pv"],
                title="Forecasted Power Generation Comparison",
                labels={
                    "power_kw": f"Forecast with {inverter_type}",
                    "power_kw_no_live_pv": "Forecast without recent PV data"
                }
            )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Power (kW)",
                legend_title="Forecast Type",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display raw data
            st.subheader("Raw Forecast Data")
            predictions_df_display = predictions_df.reset_index().rename(columns={'index': 'Date'})
            predictions_df_display = predictions_df_display.set_index('Date')
            st.dataframe(predictions_df_display, use_container_width=True)

# Some information about the app
st.sidebar.info(
    """
    This dashboard runs
    [Open Climate Fix](https://openclimatefix.org/)'s
    
    [Open Source Quartz Solar Forecast](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/).
    
    Click 'Run Forecast' and add the Home-Owner approved authentication URL to see the results.
    """
)

# Footer
st.markdown("---")
st.markdown(f"Created with ❤️ by [Open Climate Fix](https://openclimatefix.org/)")