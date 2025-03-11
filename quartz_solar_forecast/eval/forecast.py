import os
import pandas as pd
import xarray as xr
#from psp.data_sources.nwp import NwpDataSource
#from psp.data_sources.pv import NetcdfPvDataSource
from psp.serialization import load_model
#from psp.typings import X

from quartz_solar_forecast.data import get_nwp, make_pv_data
from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.forecasts.v1 import forecast_v1
#from quartz_solar_forecast.data import format_nwp_data

from datetime import datetime

class Forecaster:
    """
    A class to handle solar forecasting efficiently by loading the model once
    and providing methods to run forecasts on multiple inputs.
    """
    
    def __init__(self, model_path = None):
        """
        model_path: Path to the model file. If None, uses the default path.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if model_path is None:
            model_path = f"{dir_path}/../models/model-0.3.0.pkl"
        
        # Load model once during initialization
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def forecast_single(self, pv_row, nwp_site_df, nwp_source = "ICON"):
        """
        pv_row: DataFrame row containing PV site data
        nwp_site_df: DataFrame with NWP data for the site
        nwp_source: Source of NWP data (default: "ICON")
        """
        site = PVSite(
            latitude = pv_row["latitude"],
            longitude = pv_row["longitude"],
            capacity_kwp = pv_row["capacity"],
        )
        
        pv_id = pv_row["pv_id"]
        ts = pv_row["timestamp"]
        
        nwp_df_copy = nwp_site_df.copy()
        for c in ["timestamp", "latitude", "longitude", "pv_id"]:
            if c in nwp_df_copy.columns:
                nwp_df_copy = nwp_df_copy.drop(columns = c)
        
        nwp_df_copy.set_index("time", inplace = True, drop = True)
        
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
            
        # Preparing data for model
        nwp_xr = format_nwp_data(df = nwp_df_copy, nwp_source = nwp_source, site = site)
        pv_xr = make_pv_data(site = site, ts = ts)
        
        # Run model
        pred_df = forecast_v1(nwp_source, nwp_xr, pv_xr, ts, model = self.model)
        
        # Processing
        pred_df = pred_df.resample("1H").mean()
        pred_df["horizon_hour"] = range(0, len(pred_df))
        pred_df["pv_id"] = pv_id
        pred_df['timestamp'] = pred_df.index
        return pred_df
    
    def forecast_batch(self, pv_df, nwp_df, nwp_source = "ICON"):
        """        
        pv_df: DataFrame with PV site data
        nwp_df: DataFrame with NWP data
        nwp_source: Source of NWP data (default: "ICON")
        """
        all_predictions = []
        
        for i in range(len(pv_df)):
            print(f"Running forecast for {i + 1} of {len(pv_df)}")
            pv_row = pv_df.iloc[i]
            
            # Filtering NWP data for current site and timestamp
            nwp_site_df = nwp_df[
                (nwp_df["pv_id"] == pv_row.pv_id) & (nwp_df["timestamp"] == pv_row.timestamp)
            ]
            
            # Run forecast for single site
            pred_df = self.forecast_single(pv_row, nwp_site_df, nwp_source)
            all_predictions.append(pred_df)
        
        # Combine all predictions
        if all_predictions:
            return pd.concat(all_predictions)
        else:
            return pd.DataFrame()


def run_forecast(pv_df: pd.DataFrame, nwp_df: pd.DataFrame, nwp_source = "ICON") -> pd.DataFrame:
    forecaster = Forecaster()
    return forecaster.forecast_batch(pv_df, nwp_df, nwp_source)