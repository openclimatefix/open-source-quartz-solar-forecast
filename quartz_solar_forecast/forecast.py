import os
import pandas as pd

from quartz_solar_forecast.data import get_nwp, make_pv_data
from quartz_solar_forecast.pydantic_models import PVSite

from datetime import datetime

from quartz_solar_forecast.forecasts.v1 import forecast_v1

def run_forecast(site: PVSite, ts: datetime | str = None, nwp_source: str = "icon") -> pd.DataFrame:
    """
    Run the forecast from NWP data

    :param site: the PV site
    :param ts: the timestamp of the site. If nothing is specified, it defaults to the current timestamp rounded down to 15 minutes.
    :param nwp_source: the nwp data source. Either "gfs" or "icon". Defaults to "icon"
    :return: The PV forecast of the site for time (ts) for 48 hours
    """

    # set timestamp to now if not provided
    if ts is None:
        ts = pd.Timestamp.now().round("15min")

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    # make pv and nwp data from GFS
    nwp_xr = get_nwp(site=site, ts=ts)
    pv_xr = make_pv_data(site=site, ts=ts)

    # load and run models
    pred_df = forecast_v1(nwp_source, nwp_xr, pv_xr, ts)

    return pred_df
