import os
from datetime import datetime
from huggingface_hub import login, HfFileSystem
import forecast_csv

if __name__ == "__main__":

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    now = datetime.utcnow()
    latitude = 51.75
    longitude = -1.25
    forecast = forecast_csv.forecast_for_site(latitude=latitude, longitude=longitude, capacity_kwp=1.25, init_time=now)

    path = now.strftime(f"data/%Y/%-m/%-d/{latitude}_{longitude}_%Y%m%d_%H.csv")
    login(hf_token)
    fs = HfFileSystem()
    with fs.open(f"datasets/{hf_repo}/{path}", "w") as f:
        forecast.to_csv(path_or_buf=f)
