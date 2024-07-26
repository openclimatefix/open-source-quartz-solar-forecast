import os
from datetime import datetime
from huggingface_hub import login, HfFileSystem
import forecast_csv

if __name__ == "__main__":

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    login(hf_token)
    fs = HfFileSystem()
    now = datetime.utcnow()
    latitude = 51.75
    longitude = -1.25
    capacity_kwp = 1.25

    for model in ["gb", "xgb"]:
        forecast = forecast_csv.forecast_for_site(latitude, longitude, capacity_kwp, model, now)

        path = now.strftime(f"data/%Y/%-m/%-d/{model}_{latitude}_{longitude}_{capacity_kwp}_%Y%m%d_%H.csv")
        with fs.open(f"datasets/{hf_repo}/{path}", "w") as f:
            forecast.to_csv(path_or_buf=f)
