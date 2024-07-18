# Streamlit Dashboard for Open Source Quartz Solar Forecast

This Streamlit Dashboard provides an interactive interface for running and visualizing solar forecasts using the [Open Source Quartz Solar Forecast](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast) model developed by [Open Climate Fix](https://openclimatefix.org/).

## Features

- Configure PV site parameters (latitude, longitude, capacity)
- Select inverter type (No Inverter or Enphase)
- Enphase API authentication flow (if applicable)
- Run solar forecast
- Visualize forecast results with interactive charts
- Compare forecasts with and without recent PV data (if applicable)
- Display raw forecast data and provide an option to download it as CSV

## How to Run

1. Clone the repository and install all the dependencies in a virtual environment on a Linux System(or WSL):
   `pip install -e .` and `pip install -r requirements.txt`

2. Set up environment variables (if applicable):

- Create a `.env` file in your root directory
- Add the following variables:
  ```
  ENPHASE_CLIENT_ID=your_client_id
  ENPHASE_CLIENT_SECRET=your_client_secret
  ENPHASE_API_KEY=your_api_key
  ENPHASE_SYSTEM_ID=your_system_id
  ```

3. Navigate to the `dashboards/dashboard_2` directory.

4. Run the Streamlit app: `streamlit run app.py`

5. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## Using the App

1. **Configure PV Site:**

- Use the sidebar to input latitude, longitude, and capacity of the PV site.
- Alternatively, check "Use Default Values" to use pre-set values.

2. **Select Inverter Type:**

- Choose between "No Inverter" and "Enphase" from the dropdown menu.

3. **Enphase Authorization (if applicable):**

- If you select Enphase, follow the authorization process:
  - Click the provided authorization URL
  - Grant permissions on the Enphase website
  - Copy the redirect URL and paste it back into the app

4. **Run Forecast:**

- Click the "Run Forecast" button to generate predictions.

5. **View Results:**

- See current power, total forecasted energy, and peak forecasted power.
- Examine the interactive line chart comparing forecasts with and without recent PV data.
- Review the raw forecast data table and optionally download it as CSV for further processing.

## Additional Information

- Forecasts are generated for the next 24 hours in 15-minute intervals.
- The app demonstrates the impact of using recent PV data (from Enphase) on forecast accuracy.