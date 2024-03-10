""" Make a script to generate forecasts for multiple sites.

The idea is to generate forecasts for multiple sites and to combine them into a single csv. 
A list with several sites containing pv_id, latitude, longitude and capacity will be passed
through the forecaster then each forecast will be saved to a csv.

"""

# imports
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datasets import load_dataset
import pandas as pd

# huggingface url
site_data = load_dataset("openclimatefix/uk_pv")

# open forecasts.csv to output
file_path = "quartz_solar_forecast/dataset/forecasts.csv"

# initialize list of pv ids
pv_ids = [
    9531
]

# initialize list of PV Sites
sites = []

# get specific information given pv id
def get_info(pv_id):
    # Filter the site data for the given pv id
    filtered_data = site_data.filter(lambda example: example['ss_id'] == pv_id)

    # If there is no matching pv id in site data
    if len(filtered_data) == 0:
        return "No data found for the given ss_id."

    # Extract the latitude, longitude and capacity
    latitude = filtered_data['latitude_rounded'][0]
    longitude = filtered_data['longitude_rounded'][0]
    capacity = filtered_data['kwp'][0]

    return [latitude, longitude, capacity]

# get site data from metadata
for pv_id in pv_ids:
    row = get_info(pv_id)
    sites.append(PVSite(row[0], row[1], row[2]))

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