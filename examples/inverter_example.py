""" Example code to run the forecast"""
import pandas as pd
from datetime import datetime
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

# Set plotly backend to be plotly, you might have to install plotly
pd.options.plotting.backend = "plotly"


def main():

    ts = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # make input data with live enphase data
    site_live = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25, inverter_type="enphase")

    # make input data with nan data
    site_no_live = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

    # run model, with and without recent pv data
    predictions_with_recent_pv_df = run_forecast(site=site_live, ts=ts)
    predictions_df = run_forecast(site=site_no_live, ts=ts) 

    predictions_with_recent_pv_df["power_kw_no_live_pv"] = predictions_df["power_kw"]

    # plot
    fig = predictions_with_recent_pv_df.plot(
        title="PV Forecast",
        template="plotly_dark",
        y=["power_kw", "power_kw_no_live_pv"],
        labels={"value": "Power (kW)", "index": "Time"},
    )
    fig.show(renderer="browser")


if __name__ == "__main__":
    main()
