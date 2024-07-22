import os

from huggingface_hub import login
from datasets import Dataset

import forecast_csv

if __name__ == "__main__":

    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    forecast = forecast_csv.forecast_for_site(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    ds = Dataset.from_pandas(forecast)
    hf_repo = os.getenv("HF_REPO")
    ds.push_to_hub(hf_repo)
