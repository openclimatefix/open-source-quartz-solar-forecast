import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import os
import requests
from PIL import Image
from dotenv import load_dotenv

from quartz_solar_forecast.pydantic_models import PVSite

# Load environment variables
load_dotenv()

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
    st.sidebar.write("Enphase Authorization")
    auth_url_data = make_api_request("/enphase/auth_url")
    if auth_url_data:
        auth_url = auth_url_data["auth_url"]
        st.sidebar.write("Please visit the following URL to authorize the application:")
        st.sidebar.markdown(f"[Enphase Authorization URL]({auth_url})")
        st.sidebar.write("After authorization, you will be redirected to a URL. Please copy the entire URL and paste it below:")
    
    enphase_redirect_url = st.sidebar.text_input("Enter the redirect URL:", key="enphase_redirect_url")

if st.sidebar.button("Run Forecast"):
    access_token = None
    enphase_system_id = None

    if inverter_type == "Enphase":
        if enphase_redirect_url:
            token_data = make_api_request("/enphase/access_token", method="POST", data={"full_auth_url": enphase_redirect_url})
            if token_data and "access_token" in token_data:
                access_token = token_data["access_token"]
                enphase_system_id = os.getenv('ENPHASE_SYSTEM_ID')
        else:
            st.error("Please enter the Enphase authorization redirect URL before running the forecast.")
            st.stop()

    if inverter_type == "Enphase" and (not access_token or not enphase_system_id):
        st.error("Enphase authorization is required. Please complete the authorization process and provide the system ID.")
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
        "access_token": access_token,
        "enphase_system_id": enphase_system_id
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
        fig = px.line(
            predictions.reset_index(),
            x="index",
            y=["power_kw", "power_kw_no_live_pv"],
            title="Forecasted Power Generation",
            labels={
                "power_kw": f"Forecast with {inverter_type}" if inverter_type != "No Inverter" else "Forecast",
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