"""Same config as the uk_pv but trains a second model for the last month.

This second model is what we use in production.
"""

import datetime as dt

import numpy as np

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import DateSplits, PvSplits, TestDateSplit, TrainDateSplit, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/uk_pv/5min_v3.nc"
NWP_DATA_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
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
    try:
        value = float(d["power"].quantile(0.99))
    except TypeError:
        # This exception happens if the data is empty.
        value = np.nan
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
            # add new satelite here
            nwp_data_source={
                "UKV": NwpDataSource(
                    NWP_DATA_PATHS,
                    coord_system=27700,
                    time_dim_name="init_time",
                    value_name="UKV",
                    y_is_ascending=False,
                    cache_dir=".nwp_cache",
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

    def get_date_splits(self, step_minutes: int = 1) -> DateSplits:
        # Train 2 models, one at the beginning of the test range, and one 1 month before the end.
        # The last one is the one we'll use in production.
        return DateSplits(
            train_date_splits=[
                TrainDateSplit(train_date=date, train_days=365 * 2, step_minutes=15)
                for date in [dt.datetime(2020, 1, 1), dt.datetime(2021, 10, 8)]
            ],
            test_date_split=TestDateSplit(
                start_date=dt.datetime(2020, 1, 1),
                end_date=dt.datetime(2021, 11, 8),
                step_minutes=step_minutes,
            ),
        )
