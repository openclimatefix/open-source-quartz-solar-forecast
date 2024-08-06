import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import os
import requests
from PIL import Image
from dotenv import load_dotenv
from streamlit import session_state as state

from quartz_solar_forecast.pydantic_models import PVSite

# Load environment variables
load_dotenv()

if 'enphase_access_token' not in state:
    state.enphase_access_token = None
if 'enphase_system_id' not in state:
    state.enphase_system_id = None

# Set up the base URL for the FastAPI server
FASTAPI_BASE_URL = "http://localhost:8000"

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

def make_api_request(endpoint, method="GET", data=None):
    try:
        url = f"{FASTAPI_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return None

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

inverter_type = st.sidebar.selectbox("Select Inverter", ["No Inverter", "Enphase", "Solis", "GivEnergy"])

if inverter_type == "Enphase":
    if state.enphase_access_token is None:
        st.sidebar.write("Enphase Authorization")
        auth_url_data = make_api_request("/enphase/auth_url")
        if auth_url_data:
            auth_url = auth_url_data["auth_url"]
            st.sidebar.write("Please visit the following URL to authorize the application:")
            st.sidebar.markdown(f"[Enphase Authorization URL]({auth_url})")
            st.sidebar.write("After authorization, you will be redirected to a URL. Please copy the entire URL and paste it below:")
        
        enphase_redirect_url = st.sidebar.text_input("Enter the redirect URL:", key="enphase_redirect_url")
    else:
        st.sidebar.success("Enphase is authorized.")

if st.sidebar.button("Authorize and Run Forecast"):
    if inverter_type == "Enphase":
        if enphase_redirect_url:
            # Get access token
            token_data = make_api_request("/enphase/access_token", method="POST", data={"full_auth_url": enphase_redirect_url})
            if token_data and "access_token" in token_data:
                state.enphase_access_token = token_data["access_token"]
                state.enphase_system_id = os.getenv('ENPHASE_SYSTEM_ID')
                st.sidebar.success("Enphase authorized successfully!")
            else:
                st.error("Failed to obtain Enphase access token.")
                st.stop()
        else:
            st.error("Please enter the Enphase authorization redirect URL.")
            st.stop()

    # Create PVSite object with user-input or default values
    site = PVSite(
        latitude=latitude,
        longitude=longitude,
        capacity_kwp=capacity_kwp,
        inverter_type=inverter_type.lower() if inverter_type != "No Inverter" else ""
    )

    # Prepare data for API request
    data = {
        "site": site.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nwp_source": "icon",
        "access_token": state.enphase_access_token if inverter_type == "Enphase" else None,
        "enphase_system_id": state.enphase_system_id if inverter_type == "Enphase" else None
    }

    # Make the API request
    forecast_data = make_api_request("/forecast/", method="POST", data=data)

    if forecast_data:
        st.success("Forecast completed successfully!")

        # Display current timestamp
        st.subheader(f"Forecast generated at: {forecast_data['timestamp']}")

        # Create three columns
        col1, col2, col3 = st.columns(3)

        predictions = pd.DataFrame(forecast_data['predictions'])
        
        # Ensure 'index' column exists and is of datetime type
        if 'index' not in predictions.columns:
            predictions['index'] = pd.to_datetime(predictions.index)
        else:
            predictions['index'] = pd.to_datetime(predictions['index'])
        
        predictions.set_index('index', inplace=True)
        
        with col1:
            current_power = predictions['power_kw'].iloc[-1]
            st.metric("Current Power", f"{current_power:.2f} kW")

        with col2:
            total_energy = predictions['power_kw'].sum() * 0.25  # Assuming 15-minute intervals
            st.metric("Total Forecasted Energy", f"{total_energy:.2f} kWh")

        with col3:
            peak_power = predictions['power_kw'].max()
            st.metric("Peak Forecasted Power", f"{peak_power:.2f} kW")

        # Create a line chart of power generation
        if inverter_type == "No Inverter":
            fig = px.line(
                predictions.reset_index(),
                x="index",
                y="power_kw",
                title="Forecasted Power Generation",
                labels={
                    "power_kw": "Forecast without recent PV data",
                    "index": "Time"
                }
            )
        else:
            # If an inverter is selected, we assume both 'power_kw' and 'power_kw_no_live_pv' exist
            if 'power_kw_no_live_pv' not in predictions.columns:
                st.error("Expected 'power_kw_no_live_pv' column is missing. Please check the API response.")
            else:
                fig = px.line(
                    predictions.reset_index(),
                    x="index",
                    y=["power_kw", "power_kw_no_live_pv"],
                    title="Forecasted Power Generation",
                    labels={
                        "power_kw": f"Forecast with {inverter_type}",
                        "power_kw_no_live_pv": "Forecast without recent PV data",
                        "index": "Time"
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
        st.dataframe(predictions, use_container_width=True)

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