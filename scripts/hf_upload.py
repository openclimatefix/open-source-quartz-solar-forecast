""" This script is used to run the forecast every day at 9am

Github actions runs this script every day at 9am to generate the forecast for the next 48 hours.
The results are saved to HF -
https://huggingface.co/openclimatefix/open-source-quartz-solar-forecast/tree/main/data

This means we can start to compare the difference between the forecasts and the actual generation.
"""

import os
from datetime import datetime
from huggingface_hub import login, HfFileSystem, HfApi
from quartz_solar_forecast.utils.forecast_csv import forecast_for_site
from quartz_solar_forecast.utils.file_path import get_file_path


if __name__ == "__main__":

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    print(hf_repo)

    login(hf_token)
    fs = HfFileSystem(token=hf_token)
    now = datetime.utcnow()
    latitude = 51.59
    longitude = -1.89
    capacity_kwp = 4

    for model in ["gb", "xgb"]:
        forecast = forecast_for_site(latitude, longitude, capacity_kwp, model, now)

        path = get_file_path(latitude, longitude, capacity_kwp, model, now)
        with fs.open(f"{hf_repo}/{path}", "w") as f:
            forecast.to_csv(path_or_buf=f)
