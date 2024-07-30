import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import sys
import os
import requests
from PIL import Image

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

API_URL = "http://localhost:8000"

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
    if "access_token" not in st.session_state:
        auth_url_response = requests.get(f"{API_URL}/enphase/auth_url/")
        auth_url = auth_url_response.json()["auth_url"]
        st.write("Please visit the following URL to authorize the application:")
        st.markdown(f"[Enphase Authorization URL]({auth_url})")
        st.write("After authorization, you will be redirected to a URL. Please copy the entire URL and paste it below:")
        redirect_url = st.text_input("Enter the redirect URL:")
        if redirect_url and "?code=" in redirect_url:
            auth_code = redirect_url.split("?code=")[1]
            try:
                access_token_response = requests.post(f"{API_URL}/enphase/access_token/", json={"auth_code": auth_code})
                access_token = access_token_response.json()["access_token"]
                st.session_state["access_token"] = access_token
                st.success("Authorization successful!")
            except Exception as e:
                st.error(f"Error getting access token: {str(e)}")
    else:
        st.success("Enphase authorization already completed.")

if st.sidebar.button("Run Forecast"):
    site = {
        "latitude": latitude,
        "longitude": longitude,
        "capacity_kwp": capacity_kwp,
        "inverter_type": inverter_type.lower()
    }
    
    if inverter_type == "No Inverter":
        response = requests.post(f"{API_URL}/forecast/no_inverter/", json=site)
    elif inverter_type == "Enphase":
        if "access_token" not in st.session_state:
            st.error("Enphase authorization is required. Please complete the authorization process.")
        else:
            response = requests.post(
                f"{API_URL}/forecast/enphase/",
                json={**site, "access_token": st.session_state["access_token"], "system_id": os.getenv("ENPHASE_SYSTEM_ID")}
            )
    elif inverter_type == "Solis":
        response = requests.post(f"{API_URL}/forecast/solis/", json=site)
    elif inverter_type == "GivEnergy":
        response = requests.post(f"{API_URL}/forecast/givenergy/", json=site)
    
    if response.status_code == 200:
        predictions_df = pd.DataFrame(response.json())
        ts = datetime.now(timezone.utc)

        st.success("Forecast completed successfully!")

        # Display current timestamp
        st.subheader(f"Forecast generated at: {ts}")

        # Create three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Power", f"{predictions_df['power_kw'].iloc[-1]:.2f} kW")

        with col2:
            total_energy = predictions_df["power_kw"].sum() * 0.25  # Assuming 15-minute intervals
            st.metric("Total Forecasted Energy", f"{total_energy:.2f} kWh")

        with col3:
            peak_power = predictions_df["power_kw"].max()
            st.metric("Peak Forecasted Power", f"{peak_power:.2f} kW")

        # Create a line chart of power generation
        fig = px.line(
            predictions_df.reset_index(),
            x="index",
            y="power_kw",
            title="Forecasted Power Generation",
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power (kW)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display raw data
        st.subheader("Raw Forecast Data")
        predictions_df_display = predictions_df.reset_index().rename(columns={'index': 'Date'})
        predictions_df_display = predictions_df_display.set_index('Date')
        st.dataframe(predictions_df_display, use_container_width=True)
    else:
        st.error(f"Error running forecast: {response.text}")

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
st.markdown("Created with ❤️ by [Open Climate Fix](https://openclimatefix.org/)")