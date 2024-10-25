import os
import pandas as pd
from datetime import datetime, timedelta
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite


def generate_all_forecasts(
    init_time_freq: int,
    start: datetime,
    end: datetime,
    latitude: float,
    longitude: float,
    capacity_kwp: float,
) -> pd.DataFrame:

    all_forecasts = pd.DataFrame()

    init_time = start
    while init_time <= end:
        print(f"Running forecast for initialization time: {init_time}")
        predictions_df = forecast_for_site(
            latitude, longitude, capacity_kwp, init_time=init_time
        )
        predictions_df["forecast_init_time"] = init_time
        all_forecasts = pd.concat([all_forecasts, predictions_df])
        init_time += timedelta(hours=init_time_freq)

    return all_forecasts


def forecast_for_site(
    latitude: float,
    longitude: float,
    capacity_kwp: float,
    model: str = "gb",
    init_time: datetime = None,
) -> pd.DataFrame:

    site = PVSite(latitude=latitude, longitude=longitude, capacity_kwp=capacity_kwp)
    predictions_df = run_forecast(site=site, model=model, ts=init_time)
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={"index": "datetime"}, inplace=True)
    return predictions_df


def write_out_forecasts(
    init_time_freq,
    start_datetime,
    end_datetime,
    site_name,
    latitude,
    longitude,
    capacity_kwp,
):
    """
    Generates forecasts at specified intervals and saves them into a CSV file.

    Args:
        init_time_freq (int): The frequency in hours at which the forecasts are generated.
        start_datetime (str): The starting date and time for generating forecasts.
        end_datetime (str): The ending date and time for generating forecasts.
        site_name (str): The name of the site for which the forecasts are generated.
        latitude (float): The latitude of the PV site.
        longitude (float): The longitude of the PV site.
        capacity_kwp (float): The capacity of the PV site in kilowatts peak (kWp).
    """
    start = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    start_date = start.date()
    end = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
    end_date = end.date()
    all_forecasts = generate_all_forecasts(
        init_time_freq, start, end, latitude, longitude, capacity_kwp
    )

    output_dir = os.path.join(os.getcwd(), "csv_forecasts")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_name = f"forecast_{site_name}_{start_date}_{end_date}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    all_forecasts.to_csv(output_file_path, index=False)
    print(f"Forecasts saved to {output_file_path}")

