import datetime
import os
from huggingface_hub import login
from huggingface_hub import HfFileSystem
import forecast_csv

if __name__ == "__main__":

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    now = datetime.datetime.now()
    forecast = forecast_csv.forecast_for_site(latitude=51.75, longitude=-1.25, capacity_kwp=1.25, init_time=now)

    path = now.strftime("data/%Y/%-m/%-d/%Y%m%d_%H.csv")
    login(hf_token)
    fs = HfFileSystem()
    with fs.open(f"datasets/{hf_repo}/{path}", "w") as f:
        forecast.to_csv(path_or_buf=f)
