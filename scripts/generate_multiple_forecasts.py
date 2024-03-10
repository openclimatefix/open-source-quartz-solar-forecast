""" Make a script to generate forecasts for multiple sites.

The idea is to generate forecasts for multiple sites and to combine them into a single csv. 
A list with several sites containing pv_id, latitude, longitude and capacity will be passed
through the forecaster then each forecast will be saved to a csv.

"""

# imports
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
import pandas as pd

# open forecasts.csv to output
file_path = "quartz_solar_forecast/dataset/forecasts.csv"

# initialize list of PV Sites
sites = [
    PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
]

# run prediction on multiple sites
for site in sites:
    predictions_df = run_forecast(site=site)
    
    # write to the csv
    predictions_df.to_csv(file_path, index=True)
    
print("complete")