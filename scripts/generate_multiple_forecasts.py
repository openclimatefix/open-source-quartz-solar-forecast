"""Make a script to generate forecasts for multiple sites.

The idea is to generate forecasts for multiple sites and to combine them into a 
single csv. A list with several sites containing pv_id, latitude, longitude 
and capacity will be passed through the forecaster then each forecast will be saved 
to a csv.

"""

import sys
import pandas as pd
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

# file paths
site_data_path = "quartz_solar_forecast/dataset/metadata.csv"
output_path = "quartz_solar_forecast/dataset/forecasts.csv"


def get_pv_site_data(site_data_path, pv_ids):
    """Retrieve specific pv_site data from metadata."""
    sites = {}
    df = pd.read_csv(site_data_path)

    # iterate through pv ids
    for pv_id in pv_ids:
        row = df[df["ss_id"] == pv_id]

        if not row.empty:
            # make new site
            site = PVSite(
                latitude=row["latitude_rounded"].values[0],
                longitude=row["longitude_rounded"].values[0],
                capacity_kwp=row["kwp"].values[0],
                # Add tilt and orientation here
            )

            # add site to sites dictionary
            sites[pv_id] = site
        else:
            # skip pv_id if it isn't in metadata
            print(f"Warning: PV ID {pv_id} not found in metadata. Skipping.")

    return sites


def generate_forecasts(sites, output_path):
    """Generate forecasts for pv_sites."""
    csv_data = []

    # run forecast for each pv site
    for pv_id, site in sites.items():
        predictions_df = run_forecast(site=site)

        # add to csv data
        for index, row in predictions_df.iterrows():
            csv_row = [
                pv_id,
                site.latitude,
                site.longitude,
                site.capacity_kwp,
                index.date(),
                index.time(),
                row["power_wh"],
            ]
            csv_data.append(csv_row)

    csv_df = pd.DataFrame(
        csv_data,
        columns=[
            "pv_id",
            "latitude",
            "longitude",
            "capacity_kwp",
            "date",
            "time",
            "power_wh",
        ],
    )

    # write to csv
    csv_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    pv_ids = [int(arg) for arg in sys.argv[1:]]

    if not pv_ids:
        print("Please provide PV IDs as command-line arguments.")
        sys.exit(1)

    sites = get_pv_site_data(site_data_path, pv_ids)
    generate_forecasts(sites, output_path)
