""" Example code to run the forecast"""
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime, timedelta

def main():
    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    # for enphase testing
    # site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25, inverter_type="enphase")

    
    ts = datetime.today() - timedelta(weeks=1)
    predictions_df = run_forecast(site=site, ts=ts, nwp_source="icon")

    print(predictions_df)
    print(f"Max: {predictions_df['power_wh'].max()}")


if __name__ == "__main__":
    main()
