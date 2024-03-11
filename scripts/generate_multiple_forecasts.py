""" Make a script to generate forecasts for multiple sites.

The idea is to generate forecasts for multiple sites and to combine them into a single csv. 
A list with several sites containing pv_id, latitude, longitude and capacity will be passed
through the forecaster then each forecast will be saved to a csv.

"""

# imports
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
import pandas as pd

# metadata link
site_data = "quartz_solar_forecast/dataset/metadata.csv"

# open forecasts.csv to output
output_path = "quartz_solar_forecast/dataset/forecasts.csv"

# initialize list of pv ids
pv_ids = [
    
]

# initialize list of PV Sites
sites = {}

# get specific information given pv id
def get_info(pv_id):
    df = pd.read_csv(site_data)
    
    # extract row of pv id from metadata
    row = df[df['ss_id'] == pv_id]
    return PVSite(
        latitude=row['latitude_rounded'].values[0],
        longitude=row['longitude_rounded'].values[0],
        capacity_kwp=row['kwp'].values[0],
        # would add tilt and orientation here
    )

# get site data from metadata
for pv_id in pv_ids:
    pv_site = get_info(pv_id)
    
    # add to sites
    sites[pv_id] = pv_site

# list for csv data
csv_data = []

# run forecast on multiple sites
for pv_id in pv_ids:
    # run forecast for site
    site = sites[pv_id]
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
csv_df.to_csv(output_path, index=False)
    
print("complete")