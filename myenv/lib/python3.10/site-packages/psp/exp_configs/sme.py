import datetime as dt
import functools

import numpy as np

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

# _PREFIX = ""
PV_DATA_PATH = (
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "PV/sme/v1/no_stark/sme_no_stark.nc"
)
# stark/sme_stark.nc"
# no_stark/sme_no_stark.nc"

# _PREFIX + "/pv_v3.nc"
# PV_DATA_PATH_5MIN = _PREFIX + "/pv_v3_5min.nc"

METOFFICE_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in [2018, 2019, 2020, 2021, 2022]
]

EXC_PATH = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/"
        f"merged_zarrs/test_3_temp/excarta_{year}.zarr"
    )
    for year in [2019, 2020, 2021, 2022]
]

ECMWF_PATH = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
        f"NWP/ECMWF/uk/year_merged/{year}.zarr"
    )
    for year in [2020, 2021, 2022]
]


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            # lag_minutes=60, # Lag for sites with Stark meters
            lag_minutes=2 * 24 * 60,  # Lag for sites without Stark meters
        )

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_sources={
                "UKV": NwpDataSource(
                    METOFFICE_PATHS,
                    coord_system=27700,
                    time_dim_name="init_time",
                    value_name="UKV",
                    y_is_ascending=False,
                    # Those are the variables available in our prod environment.
                    variables=[
                        "si10",
                        "vis",
                        # "r2",
                        "t",
                        "prate",
                        # "sd",
                        "dlwrf",
                        "dswrf",
                        "hcc",
                        "mcc",
                        "lcc",
                    ],
                    tolerance="168h",
                    lag_minutes=4 * 60,
                ),
                "EXC": NwpDataSource(
                    EXC_PATH,
                    coord_system=4326,
                    x_dim_name="latitude",
                    y_dim_name="longitude",
                    time_dim_name="ts",
                    x_is_ascending=True,
                    y_is_ascending=True,
                    lag_minutes=7 * 60,
                    tolerance="168h",
                ),
                "ECMWF": NwpDataSource(
                    ECMWF_PATH,
                    coord_system=4326,
                    x_dim_name="latitude",
                    y_dim_name="longitude",
                    time_dim_name="init_time",
                    value_name="UKV",
                    x_is_ascending=True,
                    y_is_ascending=False,
                    lag_minutes=6 * 60,
                    tolerance="168h",
                ),
            },
        )

    def get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=30, num_horizons=48 * 2))

    def get_model(self, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        return RecentHistoryModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=True,
            ),
            random_state=random_state,
            normalize_features=True,
            nwp_dropout=0.1,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source, pv_split=None)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2022, 1, 1),
            test_end_date=dt.datetime(2022, 12, 31),
            # data across the board, 1 training will probably be enough.
            num_trainings=4,
            train_days=365 * 2,
            # Min date because of NWP not available at the beginning of the PV data.
            min_train_date=dt.datetime(2020, 1, 10),
        )
