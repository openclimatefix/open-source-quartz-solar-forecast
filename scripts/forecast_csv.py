import os
import pandas as pd
from datetime import datetime, timedelta
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
import unittest
from unittest.mock import patch
from datetime import datetime

def generate_forecast(init_time_freq, start_datetime, end_datetime, site_name):
    """
    Generates forecasts at specified intervals and saves them into a CSV file.

    Args:
        init_time_freq (int): The frequency in hours at which the forecasts are generated.
        start_datetime (str): The starting date and time for generating forecasts.
        end_datetime (str): The ending date and time for generating forecasts.
        site_name (str): The name of the site for which the forecasts are generated.
    """
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
    
    output_dir = os.path.join(os.getcwd(), 'csv_forecasts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    output_file_name = f"forecast_{site_name}_{start_date}_{end_date}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)

    all_forecasts.to_csv(output_file_path, index=False)
    print(f"Forecasts saved to {output_file_path}")

if __name__ == "__main__":
    # please change the site name, start_datetime and end_datetime as per your requirement
    generate_forecast(
        init_time_freq=6, 
        start_datetime="2024-03-10 00:00:00",
        end_datetime="2024-03-11 00:00:00",
        site_name="Test"
    )

class TestGenerateForecast(unittest.TestCase):
    def setUp(self):
        self.site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
        self.start_datetime = "2024-03-10 00:00:00"
        self.end_datetime = "2024-03-12 00:00:00"
        self.site_name = "Test"
        self.init_time_freq = 6

    @patch('__main__.run_forecast')
    def test_generate_forecast(self, mock_run_forecast):
        mock_df = pd.DataFrame({
            'datetime': [
                datetime(2024, 3, 10, 0, 0),
                datetime(2024, 3, 10, 6, 0),
                datetime(2024, 3, 10, 12, 0),
                datetime(2024, 3, 10, 18, 0),
                datetime(2024, 3, 11, 0, 0),
                datetime(2024, 3, 11, 6, 0),
                datetime(2024, 3, 11, 12, 0),
                datetime(2024, 3, 11, 18, 0),
                datetime(2024, 3, 12, 0, 0)
            ],
            'power_kw': [0.1, 0.5, 0.8, 0.6, 0.2, 0.7, 0.9, 0.4, 0.3]
        })
        mock_run_forecast.return_value = mock_df

        generate_forecast(self.init_time_freq, self.start_datetime, self.end_datetime, self.site_name)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_subdir = 'csv_forecasts'
        output_dir = os.path.join(base_dir, output_subdir)
        start_date = datetime.strptime(self.start_datetime, "%Y-%m-%d %H:%M:%S").date()
        end_date = datetime.strptime(self.end_datetime, "%Y-%m-%d %H:%M:%S").date()
        output_file_name = f"forecast_{self.site_name}_{start_date}_{end_date}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)

        self.assertTrue(os.path.exists(output_file_path))
        os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()
