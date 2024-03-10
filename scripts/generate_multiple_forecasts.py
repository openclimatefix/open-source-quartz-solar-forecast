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

# list for csv data
csv_data = []

# run forecast on multiple sites
for site in sites:
    # run forecast for site
    predictions_df = run_forecast(site=site)
    
    # iterate through forecast rows
    for index, row in predictions_df.iterrows():
        # create list of site info and forecast
        csv_row = [site.latitude, site.longitude, site.capacity_kwp,
                   index.date(), index.time(), row['power_wh']]
        csv_data.append(csv_row)

# create dataframe for csv data
csv_df = pd.DataFrame(csv_data, columns=['latitude', 'longitude', 
                                       'capacity_kwp', 'date', 
                                       'time', 'power_wh'])

# write to csv
csv_df.to_csv(file_path, index=False)
    
print("complete")