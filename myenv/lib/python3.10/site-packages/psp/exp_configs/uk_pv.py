"""Config used to train a model based on the `uk_pv` dataset."""

import datetime as dt

import numpy as np

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.data_sources.satellite import SatelliteDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

# import multiprocessing
# import xgboost as xgb

PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/uk_pv/5min_v3.nc"
# NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"
NWP_DATA_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in range(2018, 2022)
]

SATELLITE_DATA_PATHS = [
    (
        f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year}_nonhrv.zarr"
    )
    for year in range(2018, 2022)
]

# A list of SS_ID that don't contain enough data.
# I just didn't want to calculate them everytime.
# TODO Get rid of those when we prepare the dataset.
SKIP_SS_IDS = [
    str(x)
    for x in [
        8440,
        16718,
        8715,
        17073,
        9108,
        9172,
        10167,
        10205,
        10207,
        10278,
        26778,
        26819,
        10437,
        10466,
        26915,
        10547,
        26939,
        26971,
        10685,
        10689,
        2638,
        2661,
        2754,
        2777,
        2783,
        2786,
        2793,
        2812,
        2829,
        2830,
        2867,
        2883,
        2904,
        2923,
        2947,
        2976,
        2989,
        2999,
        3003,
        3086,
        3118,
        3123,
        3125,
        3264,
        3266,
        3271,
        3313,
        3334,
        3470,
        3502,
        11769,
        11828,
        11962,
        3772,
        11983,
        3866,
        3869,
        4056,
        4067,
        4116,
        4117,
        4124,
        4323,
        4420,
        20857,
        4754,
        13387,
        13415,
        5755,
        5861,
        5990,
        6026,
        6038,
        6054,
        14455,
        6383,
        6430,
        6440,
        6478,
        6488,
        6541,
        6548,
        6560,
        14786,
        6630,
        6804,
        6849,
        6868,
        6870,
        6878,
        6901,
        6971,
        7055,
        7111,
        7124,
        7132,
        7143,
        7154,
        7155,
        7156,
        7158,
        7201,
        7237,
        7268,
        7289,
        7294,
        7311,
        7329,
        7339,
        7379,
        7392,
        7479,
        7638,
        7695,
        7772,
        15967,
        7890,
        16215,
        # This one has funny night values.
        7830,
    ]
]


def _get_capacity(d):
    # Use 0.99 quantile over the history window, fallback on the capacity as defined
    # in the metadata.
    value = float(d["power"].quantile(0.99))
    if not np.isfinite(value):
        value = float(d.coords["capacity"].values)
    return value


def _get_tilt(d):
    tilt_values = d["tilt"].values
    return tilt_values


def _get_orientation(d):
    orientation_values = d["orientation"].values
    return orientation_values


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power", "kwp": "capacity"},
            ignore_pv_ids=SKIP_SS_IDS,
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_sources={
                "UKV": NwpDataSource(
                    NWP_DATA_PATHS,
                    coord_system=27700,
                    time_dim_name="init_time",
                    value_name="UKV",
                    y_is_ascending=False,
                    # cache_dir=".nwp_cache",
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
                ),
            },
            satellite_data_sources={
                "EUMETSAT": SatelliteDataSource(
                    SATELLITE_DATA_PATHS,
                    x_is_ascending=False,
                ),
            },
        )

    def _get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=48 * 4))

    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        kwargs = self.get_data_source_kwargs()
        return RecentHistoryModel(
            config=self._get_model_config(),
            **kwargs,
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=True,
                #
                # We have done some tests with xgboost and keep this as an example but note that we
                # haven't added xgboost to our list of dependencies.
                #
                # sklearn_regressor=xgb.XGBRegressor(
                #     objective='reg:pseudohubererror',
                #     eta=0.1,
                #     n_estimators=200,
                #     max_depth=5,
                #     min_child_weight=20,
                #     tree_method='hist',
                #     n_jobs=multiprocessing.cpu_count() // 2,
                # ),
            ),
            random_state=random_state,
            normalize_features=True,
            capacity_getter=_get_capacity,
            tilt_getter=_get_tilt,
            orientation_getter=_get_orientation,
            pv_dropout=0.1,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2020, 1, 1),
            test_end_date=dt.datetime(2021, 11, 8),
            train_days=356 * 2,
        )
