import datetime as dt

from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.yesterday import YesterdayPvSiteModel
from psp.typings import Horizons

PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/mone/pv_v3.nc"


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=NetcdfPvDataSource(PV_DATA_PATH),
        )

    def get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=30, num_horizons=36 * 2))

    def get_model(self, **kwargs) -> PvSiteModel:
        return YesterdayPvSiteModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            # Same window as the duration of the horizons.
            window_minutes=30,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source, pv_split=None)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2021, 7, 1),
            test_end_date=dt.datetime(2022, 12, 31),
            # Using 3 trainings because the NWP data situation changes over time. When we have NWP
            # data across the board, 1 training will probably be enough.
            num_trainings=1,
            train_days=2,
        )
