from datetime import timedelta

import numpy as np
import xarray as xr

from psp.data_sources.pv import PvDataSource
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.typings import Features, Timestamp, X, Y


class YesterdayPvSiteModel(PvSiteModel):
    """Baseline that returns the power output of the previous day at the same time."""

    def __init__(self, config: PvSiteModelConfig, pv_data_source: PvDataSource, window_minutes=30):
        self.set_data_sources(pv_data_source)
        self._win = timedelta(minutes=window_minutes)
        super().__init__(config)

    def set_data_sources(self, pv_data_source: PvDataSource):
        self._pv_data_source = pv_data_source

    def predict_from_features(self, x: X, features: Features) -> Y:
        powers = features["yesterday_means"]
        assert isinstance(powers, np.ndarray)
        return Y(powers=powers)

    def get_features(self, x: X, is_training: bool = False) -> Features:
        data_source = self._pv_data_source.as_available_at(x.ts)
        max_minutes = max(x[1] for x in self.config.horizons)

        yesterday = x.ts - timedelta(days=1)

        start = yesterday - self._win / 2
        end = yesterday + timedelta(minutes=max_minutes) + self._win / 2

        data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=start,
            end_ts=end,
        )["power"]

        powers = [
            self._get_features_for_one_ts(
                # Modulo so that if we are trying to predict 27 hours in advance, we'll get the same
                # thing as 3 hours in advance (which has the same "yesterday" value).
                data,
                yesterday + timedelta(minutes=((start + end) // 2) % (24 * 60)),
            )
            for [start, end] in self.config.horizons
        ]

        return dict(
            yesterday_means=np.array(powers),
        )

    def _get_features_for_one_ts(self, data: xr.DataArray, ts: Timestamp) -> float:
        start = ts - self._win / 2
        end = ts + self._win / 2

        da = data.sel(ts=slice(start, end))

        if da.size == 0:
            return np.nan
        else:
            return float(da.mean())
