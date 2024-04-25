from datetime import datetime

import pandas as pd
import pickle

from quartz_solar_forecast.data import get_nwp, make_pv_data
from quartz_solar_forecast.forecasts import forecast_v1, TryolabsSolarPowerPredictor
from quartz_solar_forecast.pydantic_models import PVSite
from psp.models.recent_history import RecentHistoryModel
from xgboost.sklearn import XGBRegressor

def predict_ocf(
    site: PVSite, model=None, ts: datetime | str = None, nwp_source: str = "icon"
):
    """Run the forecast with the OCF model"""
    if ts is None:
        ts = pd.Timestamp.now().round("15min")

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    # make pv and nwp data from nwp_source
    nwp_xr = get_nwp(site=site, ts=ts, nwp_source=nwp_source)
    pv_xr = make_pv_data(site=site, ts=ts)

    # load and run models
    pred_df = forecast_v1(nwp_source, nwp_xr, pv_xr, ts, model=model)

    return pred_df


def predict_tryolabs(
    site: PVSite, model=None, ts: datetime | str = None, nwp_source: str = "icon"
):
    """Run the forecast with the tryolabs model"""
    solar_power_predictor = TryolabsSolarPowerPredictor(model=model)

    if ts is None:
        start_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_time = pd.Timestamp.now().floor("15min")
    else:
        start_time = pd.to_datetime(ts)

    end_time = start_time + pd.Timedelta(hours=48)

    predictions = solar_power_predictor.predict_power_output(
        latitude=site.latitude,
        longitude=site.longitude,
        start_date=start_date,
        kwp=site.capacity_kwp,
        orientation=site.orientation,
        tilt=site.tilt,
    )

    predictions = predictions[
        (predictions["date"] >= start_time) & (predictions["date"] < end_time)
    ]
    predictions = predictions.reset_index(drop=True)
    predictions.set_index("date", inplace=True)

    return predictions


def check_model_file_is_tryolabs_model(obj: object) -> bool:
    """Check if the model file is a tryolabs model"""
    return isinstance(obj, XGBRegressor)


def check_model_file_is_ocf_model(obj: object) -> bool:
    """Check if the model file is an OCF model"""
    return isinstance(obj, tuple) and obj[0] == RecentHistoryModel and obj[1] is dict


def predict_solar_power(
    site: PVSite,
    model: str = None,
    ts: datetime | str = None,
    nwp_source: str = "icon",
) -> pd.DataFrame:
    """
    Predict solar power output for a given site using a specified model.

    :param site: the PV site
    :param model: the model to use for prediction
    :param ts: the timestamp of the site. If None, defaults to the current timestamp rounded down to 15 minutes.
    :param nwp_source: the nwp data source. Either "gfs" or "icon". Defaults to "icon"
    :return: The PV forecast of the site for time (ts) for 48 hours
    """
    if not model:
        return predict_ocf(site=site, model=None, ts=ts, nwp_source=nwp_source)

    # check file path is pkl
    if not model.endswith(".pkl"):
        raise ValueError("Model file must be a .pkl file")

    with open(model, "rb") as f:
        model = pickle.load(f)

    if check_model_file_is_tryolabs_model(model):
        return predict_tryolabs(site, model, ts, nwp_source)

    if check_model_file_is_ocf_model(model):
        return predict_ocf(site, model, ts, nwp_source)

    raise ValueError(f"Unsupported model type: {type(model)}")
