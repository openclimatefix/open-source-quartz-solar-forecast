import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.typings import Features, X, Y


class HistoricalForecasts(PvSiteModel):
    """Wrap historical forecasts into a model.

    This way we can easily compare historical forecasts from clients with our models, using the same
    code as usual (train_model.py, eval_model.py scripts, notebooks, etc.).
    """

    def __init__(self, config: PvSiteModelConfig, data_or_path: pathlib.Path | str | xr.Dataset):
        """
        Arguments:
        ---------
        data_or_path: Xarray data or path to data. We expect this to be a `xarray.Dataset` with
            dimensions `("pv_id", "time", "step")` and one "power" data variable. We assume that the
            forecast was done at time "time" and that the "power" is the prediction for the average
            power between "step" and the next "step".
        """
        super().__init__(config)

        if not isinstance(data_or_path, xr.Dataset):
            data_or_path = xr.open_dataset(data_or_path)

        self._data = data_or_path

    def predict_from_features(self, x: X, features: Features) -> Y:
        data = self._data.sel(pv_id=x.pv_id)
        # Get the most latest forecast available at `x.ts`.
        data = data.sel(time=x.ts, method="ffill")

        # The time at which that forecast was made.
        time = data.coords["time"].values

        min_step = data.coords["step"].min().values
        max_step = data.coords["step"].max().values

        powers = np.empty(len(self.config.horizons)) * np.nan

        for i, (h_start, _) in enumerate(self.config.horizons):
            step_py = (x.ts + dt.timedelta(minutes=h_start)) - pd.to_datetime(time).to_pydatetime()
            # Make the time delta into a numpy time delta. This is all very painful!
            step = np.timedelta64(int(step_py.total_seconds()), "s")
            if min_step <= step <= max_step:
                powers[i] = data.sel(step=step)["power"].values

        return Y(powers=powers)

    def get_features(self, x: X, is_training: bool = False) -> Features:
        return {}
