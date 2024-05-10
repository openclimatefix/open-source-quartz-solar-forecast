from datetime import datetime

import pandas as pd
import zipfile
import gdown
import os.path

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
    site: PVSite, model=None, ts: datetime | str = None):#, nwp_source: str = "icon"
#):
    """Run the forecast with the tryolabs model"""
    solar_power_predictor = TryolabsSolarPowerPredictor(model=model)

    if ts is None:
        start_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_time = pd.Timestamp.now().floor("15min")
    else:
        start_date = pd.Timestamp(ts).strftime("%Y-%m-%d")
        start_time = pd.Timestamp(ts).floor("15min")
  
    end_time = start_time + pd.Timedelta(hours=48)

    predictions = solar_power_predictor.predict_power_output(
        latitude=site.latitude,
        longitude=site.longitude,
        start_date=start_date,
        kwp=site.capacity_kwp,
        orientation=site.orientation,
        tilt=site.tilt,
    )

    if predictions is not None:
        predictions = predictions[
            (predictions["date"] >= start_time) & (predictions["date"] < end_time)
        ]
        predictions = predictions.reset_index(drop=True)
        predictions.set_index("date", inplace=True)
        print("Predictions finished.")
        return predictions


def download_model(filename, file_id):
    """
    Download model from google drive.

    Parameters
    ----------
    filename : str
        The name of the model to be saved
    file_id: 
        Google id of the model file
    """
    gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)


def decompress_zipfile(filename: str):
    """
    Extract all files contained in a .zip file to the current directory.
    filename must contain .zip extension

    Parameters
    ----------
    filename : str
        The name of the .zip file to be decompressed
    """
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall()


def run_forecast(
    site: PVSite,
    model: str = "tryolabs",
    ts: datetime | str = None,
    nwp_source: str = "icon",
) -> pd.DataFrame:
    """
    Predict solar power output for a given site using a specified model.

    :param site: the PV site
    :param model: the model to use for prediction, choose between "ocf" and "tryolabs",
                    by default "tryolabs" is used
    :param ts: the timestamp of the site. If None, defaults to the current timestamp rounded down to 15 minutes.
    :param nwp_source: the nwp data source. Either "gfs" or "icon". Defaults to "icon" 
                       (only relevant if model=="ocf")
    :return: The PV forecast of the site for time (ts) for 48 hours
    """

    if model == "ocf":
        return predict_ocf(site, None, ts, nwp_source)
              
    if model == "tryolabs":
        
        model_file = "model_10_1.ubj"
        file_id = "1PIriCDVkz7-y2qnt7GJYyZ0ToAGpgPXb"
        zipfile_model = model_file + ".zip"

        if not os.path.isfile(zipfile_model):
            print("Downloading model ...")
            download_model(zipfile_model, file_id)
        if not os.path.isfile(model_file):
            print("Preparing model ...")
            decompress_zipfile(zipfile_model)
        print("Loading model ...")
        loaded_model = XGBRegressor()
        loaded_model.load_model(model_file)
        print("Making predictions ...")
       
        return predict_tryolabs(site, loaded_model, ts)
      
    raise ValueError(f"Unsupported model: {model}. Choose between 'tryolabs' and 'ocf'")
