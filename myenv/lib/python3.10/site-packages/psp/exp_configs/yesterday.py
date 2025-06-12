import functools

from psp.data_sources.pv import NetcdfPvDataSource
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.yesterday import YesterdayPvSiteModel
from psp.typings import Horizons

PV_DATA_PATH = "data/5min_2.netcdf"


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(PV_DATA_PATH)

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(data_source=self.get_pv_data_source())

    @functools.cache
    def _get_model_config(self):
        return PvSiteModelConfig(
            horizons=Horizons(
                duration=15,
                num_horizons=48 * 4,
            )
        )

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return YesterdayPvSiteModel(self._get_model_config(), **self.get_data_source_kwargs())
