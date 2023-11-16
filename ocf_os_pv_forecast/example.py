""" Example code to run the forecast"""
from datetime import datetime

from ocf_os_pv_forecast.forecast import run_forecast
from ocf_os_pv_forecast.pydantic_models import PVSite


def main():
    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    ts = datetime(2023, 10, 30, 0, 0, 0)

    # run model
    predications_df = run_forecast(site=site, ts=ts)

    print(predications_df)
    print(f"Max: {predications_df['power_wh'].max()}")


if __name__ == "__main__":
    main()
