"""Main config for the "island" use-case."""

import datetime as dt

from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.historical_forecasts import HistoricalForecasts
from psp.typings import Horizons

_PREFIX = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island"
PV_TARGET_DATA_PATH = _PREFIX + "/pv_hourly_v6.nc"
HISTORY = _PREFIX + "/historical_predictions.nc"


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_TARGET_DATA_PATH,
        )

    def get_data_source_kwargs(self):
        return {}

    def get_model_config(self):
        return PvSiteModelConfig(
            horizons=Horizons(
                duration=60,
                num_horizons=48,
            )
        )

    def get_model(self, **kwargs) -> PvSiteModel:
        return HistoricalForecasts(self.get_model_config(), data_or_path=HISTORY)

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(
            pv_data_source,
            pv_split=None,
        )

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2022, 1, 1),
            test_end_date=dt.datetime(2022, 10, 14),
            train_days=1,
            step_minutes=60,
        )
