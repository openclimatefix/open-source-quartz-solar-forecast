import os
import pandas as pd
from datetime import datetime, timedelta
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

def generate_forecast(init_time_freq, start_datetime, end_datetime, site_name):
    start = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    start_date = start.date()
    end = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
    end_date = end.date()

    all_forecasts = pd.DataFrame()
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

    init_time = start
    while init_time <= end:
        print(f"Running forecast for initialization time: {init_time}")
        predictions_df = run_forecast(site=site, ts=init_time.strftime("%Y-%m-%d %H:%M:%S"))
        predictions_df['forecast_init'] = init_time
        all_forecasts = pd.concat([all_forecasts, predictions_df])
        init_time += timedelta(hours=init_time_freq)
    
    base_dir = os.path.dirname(os.path.realpath(__file__))

    output_subdir = 'csv_forecasts'



    output_dir = os.path.join(base_dir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    output_file_name = f"forecast_{site_name}_{start_date}_{end_date}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)


    all_forecasts.to_csv(output_file_path, index=False)
    print(f"Forecasts saved to {output_file_path}")

# please change the site name, start_datetime and end_datetime as per your requirement
generate_forecast(
    init_time_freq=6, 
    start_datetime="2023-11-01 00:00:00",
    end_datetime="2023-11-02 00:00:00",
    site_name="Test"
)
