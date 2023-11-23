from datetime import datetime

import pandas as pd
from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource
from psp.serialization import load_model
from psp.typings import X

from ocf_os_pv_forecast.data import get_gfs_nwp, make_pv_data
from ocf_os_pv_forecast.pydantic_models import PVSite

from datetime import datetime


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
    nwp_xr = get_gfs_nwp(site=site, ts=ts)
    pv_xr = make_pv_data(site=site, ts=ts)

    # load model, TODO move locally
    model = load_model("s3://pvsite-ml-models-development/models/model-0.3.0.pkl")

    # format pv and nwp data
    pv_data_source = NetcdfPvDataSource(
        pv_xr,
        id_dim_name="pv_id",
        timestamp_dim_name="timestamp",
        rename={"generation_wh": "power", "kwp": "capacity"},
        ignore_pv_ids=[],
    )
    nwp = NwpDataSource(nwp_xr, value_name="gfs")
    model.set_data_sources(pv_data_source=pv_data_source, nwp_data_sources={"GFS": nwp})

    # make prediction
    x = X(pv_id="1", ts=ts)
    pred = model.predict(x)

    # format into timerange and put into pd dataframe
    times = pd.date_range(start=x.ts, periods=len(pred.powers), freq="15min")
    pred_df = pd.DataFrame({"power_wh": pred.powers}, index=times)

    return pred_df
