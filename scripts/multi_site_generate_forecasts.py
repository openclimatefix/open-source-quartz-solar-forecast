from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
import pandas as pd


def generate_forecasts(sites_info, forecast_date):
    """Generate forecasts for multiple PV sites.

    This function takes a list of site information tuples and a forecast date as input. For each site, it creates a PVSite object,
    runs the forecast using the `run_forecast` function from the `quartz_solar_forecast` module, and generates a DataFrame
    containing the site's latitude, longitude, capacity, and power forecast values. Finally, it concatenates all the site
    DataFrames into a single DataFrame and returns it.

    Args:
        sites_info (list): List of tuples containing site information. Each tuple should be in the format
            (pv_id, latitude, longitude, capacity). Latitude and longitude are geographic coordinates, and capacity
            is the site's capacity in kilowatts peak (kWp).
        forecast_date (str): Date for which the forecast is generated (format: "YYYY-MM-DD").

    Returns:
        pandas.DataFrame: DataFrame containing forecasts for each PV site. The DataFrame has columns for each site's
            latitude, longitude, capacity, and power forecast, with a column name in the format "{pv_id} Power".
            The index of the DataFrame is set to the forecast dates.
    """
    all_forecasts = []  # List to store DataFrames for each site

    # Loop through each site information
    for site_info in sites_info:
        # Unpack site information from the tuple
        pv_id, latitude, longitude, capacity = site_info

        # Create PVSite object for the site
        site = PVSite(latitude=latitude, longitude=longitude, capacity_kwp=capacity)

        # Run forecast for the site
        forecast = run_forecast(site=site, ts=forecast_date)

        # Flatten forecast values to a 1D array
        forecast_values = forecast.values.flatten()

        # Extract forecast dates from the forecast index
        forecast_dates = forecast.index

        # Create a DataFrame for the current site
        site_df = pd.DataFrame(
            {
                f"{pv_id}_latitude": [latitude]
                * len(forecast_dates),  # Repeat latitude for each forecast date
                f"{pv_id}_longitude": [longitude]
                * len(forecast_dates),  # Repeat longitude for each forecast date
                f"{pv_id}_capacity": [capacity]
                * len(forecast_dates),  # Repeat capacity for each forecast date
                f"{pv_id} Power": forecast_values,  # Power forecast column with renamed header
            },
            index=forecast_dates,
        )

        # Append the site DataFrame to the list
        all_forecasts.append(site_df)

    # Concatenate all site DataFrames into a single DataFrame along the column axis
    master_df = pd.concat(all_forecasts, axis=1)

    return master_df


if __name__ == "__main__":
    # Example input for multiple PV sites
    sites_info = [
        (
            "Site1",
            51.75,
            -1.25,
            1.25,
        ),  # Site 1 information (pv_id, latitude, longitude, capacity)
        (
            "Site2",
            52.0,
            -1.5,
            1.5,
        ),  # Site 2 information (pv_id, latitude, longitude, capacity)
        # Add more sites as needed, with format (pv_id, latitude, longitude, capacity)
    ]
    forecast_date = "2023-11-01"  # Forecast date
    output_file = "multi_site_pv_forecasts.csv"  # Output file name

    # Generate forecasts for the given sites and forecast date
    forecasts = generate_forecasts(sites_info, forecast_date)

    # Save forecasts to a CSV file
    forecasts.to_csv(output_file)
