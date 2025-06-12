"""Config used in tests."""

import datetime as dt

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.data_sources.satellite import SatelliteDataSource
from psp.dataset import DateSplits, PvSplits, TestDateSplit, TrainDateSplit
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "psp/tests/fixtures/pv_data.nc"
NWP_PATH = "psp/tests/fixtures/nwp.zarr"
SATELLITE_PATH = "psp/tests/fixtures/satellite.zarr"


def _get_capacity(d):
    # Use 0.99 quantile over the history window, fallback on the capacity as defined
    # in the metadata.
    value = float(d["power"].quantile(0.99))
    if not np.isfinite(value):
        value = float(d.coords["factor"].values)
    return value


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_sources={
                "UKV": NwpDataSource(
                    NWP_PATH,
                    coord_system=27700,
                    time_dim_name="init_time",
                    value_name="UKV",
                    y_is_ascending=False,
                ),
            },
            satellite_data_sources={
                "EUMETSAT": SatelliteDataSource(
                    SATELLITE_PATH,
                    x_is_ascending=False,
                ),
            },
        )

    def get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=5))

    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        return RecentHistoryModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=50,
                sklearn_regressor=HistGradientBoostingRegressor(
                    random_state=np.random.RandomState(1234),
                    max_iter=10,
                ),
            ),
            random_state=random_state,
            # Make sure the NWP data is used by adding a lot of dropout on the PV data.
            pv_dropout=0.9,
            capacity_getter=_get_capacity,
            nwp_dropout=0.0,
            satellite_patch_size=0.5,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return PvSplits(
            train=["8215", "8229"],
            valid=["8215", "8229"],
            test=["8215", "8229"],
        )

    def get_date_splits(self) -> DateSplits:
        return DateSplits(
            train_date_splits=[TrainDateSplit(train_date=dt.datetime(2020, 1, 12), train_days=12)],
            test_date_split=TestDateSplit(
                start_date=dt.datetime(2020, 1, 12),
                end_date=dt.datetime(2020, 1, 15),
            ),
        )
