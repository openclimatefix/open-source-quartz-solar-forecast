

import os
import pandas as pd
from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource
from psp.serialization import load_model
from psp.typings import X

from quartz_solar_forecast.data import get_nwp, make_pv_data
from quartz_solar_forecast.pydantic_models import PVSite

from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))


def run_forecast(site: PVSite, ts: datetime | str) -> pd.DataFrame:
    """
    Run the forecast from NWP data

    :param site: the PV site
    :param ts: the timestamp of the site
    :return: The PV forecast of the site for time (ts) for 48 hours
    """

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    # make pv and nwp data from GFS
    nwp_xr = get_nwp(site=site, ts=ts)
    pv_xr = make_pv_data(site=site, ts=ts)

    # load model
    model = load_model(f"{dir_path}/models/model-0.3.0.pkl")

    # format pv and nwp data
    pv_data_source = NetcdfPvDataSource(
        pv_xr,
        id_dim_name="pv_id",
        timestamp_dim_name="timestamp",
        rename={"generation_wh": "power", "kwp": "capacity"},
        ignore_pv_ids=[],
    )
    # make NwpDataSource, get the value_name from nwp_xr itself
    nwp = NwpDataSource(nwp_xr, value_name=list(nwp_xr.data_vars)[0])
    model.set_data_sources(pv_data_source=pv_data_source, nwp_data_sources={"GFS": nwp})

    # make prediction
    x = X(pv_id="1", ts=ts)
    pred = model.predict(x)

    # format into timerange and put into pd dataframe
    times = pd.date_range(start=x.ts, periods=len(pred.powers), freq="15min")
    pred_df = pd.DataFrame({"power_wh": pred.powers}, index=times)

    return pred_df
