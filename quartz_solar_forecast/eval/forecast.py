import os
import pandas as pd
import xarray as xr
from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource
from psp.serialization import load_model
from psp.typings import X

from quartz_solar_forecast.data import get_nwp, make_pv_data
from quartz_solar_forecast.pydantic_models import PVSite

from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))


def run_forecast(pv_df: pd.DataFrame, nwp_df: pd.DataFrame, nwp_source="ICON") -> pd.DataFrame:
    """
    Run the forecast from NWP data

    :param pv_df: the PV site data. This should have columns timestamp, id, latitude, longitude, and capacity
    :param nwp_df: all the nwp data for the site and location. This shoulw have the following rows
        - timestamp: the timestamp of the site
        - temperature_2m
        - precipitation
        - shortwave_radiation
        - direct_radiation",
        - cloudcover_low",
        - cloudcover_mid",
        - cloudcover_high",
        maybe more
    """

    # load model only once
    model = load_model(f"{dir_path}/../models/model-0.3.0.pkl")

    all_predictions = []
    for i in range(len(pv_df)):

        pv_row = pv_df.iloc[i]

        site = PVSite(
            latitude=pv_row["latitude"],
            longitude=pv_row["longitude"],
            capacity_kwp=pv_row["capacity"],
        )

        nwp_site_df = nwp_df[
            (nwp_df["id"] == pv_row.pv_id) & (nwp_df["timestamp"] == pv_row.timestamp)
        ]

        pv_id = pv_df["pv_id"][i]
        ts = pv_df["timestamp"][i]

        # format
        times = nwp_site_df["time"]
        step = times - ts
        nwp_site_df = nwp_site_df.drop(columns=["id", "timestamp"])
        nwp_site_df.set_index("time", inplace=True, drop=True)

        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        # make pv and nwp data from GFS
        # TODO move this to model
        nwp_xr = xr.DataArray(
            data=nwp_site_df.values,
            dims=["step", "variable"],
            coords=dict(
                step=("step", step),
                variable=nwp_site_df.columns,
            ),
        )
        nwp_xr = nwp_xr.to_dataset(name=nwp_source)
        nwp_xr = nwp_xr.assign_coords(
            {"x": [site.longitude], "y": [site.latitude], "time": [nwp_site_df.index[0]]}
        )

        pv_xr = make_pv_data(site=site, ts=ts)

        # format pv and nwp data
        pv_data_source = NetcdfPvDataSource(
            pv_xr,
            id_dim_name="pv_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power", "kwp": "capacity"},
            ignore_pv_ids=[],
        )
        # make NwpDataSource
        nwp = NwpDataSource(paths_or_data=nwp_xr, value_name=nwp_source)
        model.set_data_sources(pv_data_source=pv_data_source, nwp_data_sources={nwp_source: nwp})

        # make prediction # TODO change '1'
        x = X(pv_id="1", ts=ts)
        pred = model.predict(x)

        # format into timerange and put into pd dataframe
        times = pd.date_range(start=x.ts, periods=len(pred.powers), freq="15min")
        pred_df = pd.DataFrame({"power_wh": pred.powers}, index=times)

        # only select hourly predictions
        pred_df = pred_df.resample("1H").mean()
        pred_df["horizon_hours"] = range(0, len(pred_df))

        all_predictions.append(pred_df)

    return pd.concat(all_predictions)
