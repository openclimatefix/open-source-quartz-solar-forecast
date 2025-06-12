import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import PvDataSource
from psp.data_sources.satellite import SatelliteDataSource
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.regressors.base import Regressor
from psp.pv import get_irradiance
from psp.typings import Batch, Features, Horizons, X, Y
from psp.utils.maths import safe_div

_log = logging.getLogger(__name__)


def to_midnight(ts: datetime) -> datetime:
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def compute_history_per_horizon(
    pv_data: xr.DataArray,
    now: datetime,
    horizons: Horizons,
) -> np.ndarray:
    """Compute per-horizon averages of PV data.

    Return:
    ------
    We return a 2d matrix where rows are our horizons and columns are days. The values are the
    average PV output for that day/horizon.
    """
    # Make sure we can fit a whole number of horizons in a day. We make this assumption in a few
    # places, in particular when rolling/resampling on the PV data history in # the
    # RecentHistory model.
    assert 24 * 60 % horizons.duration == 0

    # Treat the trivial case.
    num_values = np.prod(list(pv_data.sizes.values()))
    if num_values == 0:
        return np.empty((len(horizons), 1)) * np.nan

    df = pv_data.to_dataframe(name="value")

    # Make sure we ignore everything before `now`.
    df = df[df.index < now]

    # Resample, matching our horizons.
    df = df.resample(timedelta(minutes=horizons.duration), origin=pd.Timestamp(now)).mean()

    df = df.reset_index()

    df["date"] = df["ts"].dt.date

    df["now"] = pd.Timestamp(now)

    df["horizon_idx"] = (
        # Get the number of seconds between the date and `now`.
        (df["ts"] - df["now"]).dt.total_seconds()
        # Remove the days.
        % (24 * 60 * 60)
        # To minutes
        / 60.0
        # How many horizon durations fit in there.
        / horizons.duration
    )

    df = pd.pivot_table(
        df,
        index="horizon_idx",
        columns="date",
        values="value",
        dropna=False,
        sort=True,
    )
    df = df.reset_index(drop=True)
    df.index.rename("horizon_idx", inplace=True)

    # Add the missing horizons. Those are the ones going after 24h.
    if len(df) < len(horizons):
        df = pd.concat([df] * math.ceil(len(horizons) / len(df)), ignore_index=True)
    df = df.iloc[: len(horizons)]

    return df.to_numpy()


def minutes_since_start_of_day(ts: datetime) -> float:
    """Time of day as minutes since midnight."""
    midnight = to_midnight(ts)
    return (ts - midnight).total_seconds() / 60.0


# To maintain backward compatibility with older serialized models, we bump this version when we make
# changes to the model. We can then adapt the `RecentHistoryModel.set_state` method to take it into
# account. It's also a good idea to add a new model fixture to the `test_load_models.py` test file
# whenever we bump this, using a simplified config file like test_config1.py (to get a small model).
_VERSION = 8

# To get the metadata (tilt, orientation, capacity), we need a function that takes in
# the recent PV history and returns the value.
_MetaGetter = Callable[[xr.Dataset], float]


def _default_get_tilt(*kwargs):
    return 35.0


def _default_get_orientation(*kwargs):
    return 180.0


def _default_get_capacity(d: xr.Dataset) -> float:
    return float(d["power"].quantile(0.99))


class RecentHistoryModel(PvSiteModel):
    def __init__(
        self,
        config: PvSiteModelConfig,
        *,
        pv_data_source: PvDataSource,
        nwp_data_sources: dict[str, NwpDataSource],
        satellite_data_sources: dict[str, SatelliteDataSource] | None = None,
        regressor: Regressor,
        random_state: np.random.RandomState | None = None,
        pv_dropout: float = 0.0,
        normalize_features: bool = True,
        tilt_getter: _MetaGetter | None = None,
        orientation_getter: _MetaGetter | None = None,
        capacity_getter: _MetaGetter | None = None,
        use_capacity_as_feature: bool = True,
        num_days_history: int = 7,
        nwp_dropout: float = 0.1,
        nwp_tolerance: Optional[float] = None,
        satellite_dropout: float = 0.1,
        satellite_tolerance: Optional[float] = None,
        satellite_patch_size: float = 0.25,
        n_recent_power_values: int = 0,
    ):
        """
        Arguments:
        ---------
        pv_data_source: Pv data source.
        nwp_data_source: Nwp data source.
        regressor: The regressor to train on the features.
        random_state: Random number generator.
        pv_dropout: Probability of removing the PV data (replacing it with np.nan).
            This is only used at train-time.
        normalize_features: Should we normalize the PV-related features by
            pvlib's clearsky values.
        tilt_getter: Function to get the tilt from the PV data array.
        orientation_getter: Function to get the orientation from the PV data array.
        capacity_getter: Function to get the capacity from the PV data array.
        use_capacity_as_feature: Should we use the PV capacity as a feature.
        num_days_history: How many days to consider for the recent PV features.
        nwp_dropout: Probability of removing the NWP data (replacing it with np.nan).
            This is only used at train-time.
        nwp_tolerance: How old should the NWP predictions be before we start ignoring them.
            See `NwpDataSource.get`'s documentation for details.
        satellite_dropout: Probability of removing the satellite data (replacing it with np.nan).
        satellite_tolerance: How old should the satellite predictions be before
            we start ignoring them.
        satellite_patch_size: Size of the patch to use for the satellite data. This is in degrees.
        """
        super().__init__(config)
        # Validate some options.

        # Check if the dropout is greater than 0 for any NwpDataSource

        if nwp_dropout > 0.0 or pv_dropout > 0.0:
            assert random_state is not None

        self._pv_data_source: PvDataSource
        self._nwp_data_sources: dict[str, NwpDataSource] | None
        self._satellite_data_sources: dict[str, SatelliteDataSource] | None

        self._regressor = regressor
        self._random_state = random_state
        self._normalize_features = normalize_features
        self._pv_dropout = pv_dropout

        self._capacity_getter = capacity_getter or _default_get_capacity
        self._tilt_getter = tilt_getter or _default_get_tilt
        self._orientation_getter = orientation_getter or _default_get_orientation

        # Deprecated - keeping for backward compatibility and mypy.
        self._use_inferred_meta = None
        self._use_data_capacity = None

        self._use_capacity_as_feature = use_capacity_as_feature
        self._num_days_history = num_days_history

        self._nwp_dropout = nwp_dropout
        self._nwp_tolerance = nwp_tolerance
        self._satellite_dropout = satellite_dropout
        self._satellite_tolerance = satellite_tolerance
        self._satellite_patch_size = satellite_patch_size
        self._n_recent_power_values = n_recent_power_values

        self.set_data_sources(
            pv_data_source=pv_data_source,
            nwp_data_sources=nwp_data_sources,
            satellite_data_sources=satellite_data_sources,
        )

        # We bump this when we make backward-incompatible changes in the code, to support old
        # serialized models.
        self._version = _VERSION

        super().__init__(config)

    def set_data_sources(
        self,
        *,
        pv_data_source: PvDataSource,
        nwp_data_sources: dict[str, NwpDataSource] | None = None,
        satellite_data_sources: dict[str, SatelliteDataSource] | None = None,
    ):
        """Set the data sources.

        This has to be called after deserializing a model using `load_model`.
        """
        self._pv_data_source = pv_data_source
        self._nwp_data_sources = nwp_data_sources
        self._satellite_data_sources = satellite_data_sources

        # This ensures the nwp fixture passed for the test is a dictionary
        if isinstance(self._nwp_data_sources, dict) or self._nwp_data_sources is None:
            pass
        else:
            self._nwp_data_sources = dict(nwp_data_source=self._nwp_data_sources)

        # this make sure the satellite data is a dictionary
        if (self._satellite_data_sources is not None) and (
            not isinstance(self._satellite_data_sources, dict)
        ):
            self._satellite_data_sources = dict(satellite_data_source=self._satellite_data_sources)

        # set this attribute so it works for older models
        if not hasattr(self, "_satellite_patch_size"):
            self._nwp_patch_size = 0

    def predict_from_features(self, x: X, features: Features) -> Y:
        powers = self._regressor.predict(features)
        y = Y(powers=powers)
        return y

    def get_features(self, x: X, is_training: bool = False) -> Features:
        features = self._get_features(x, is_training=is_training)
        assert not isinstance(features, tuple)
        return features

    def _vectorize_feature(self, value: float) -> np.ndarray:
        """Take a scalar feature and make it into a vector."""
        return np.ones((len(self.config.horizons),), dtype=float) * value

    def _get_features(self, x: X, is_training: bool) -> Features:
        features: Features = dict()
        data_source = self._pv_data_source.as_available_at(x.ts)

        # We'll look at stats for the previous few days.
        history_start = to_midnight(x.ts - timedelta(days=self._num_days_history))

        # Slice as much as we can right away.
        _data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=history_start,
            end_ts=x.ts,
        )

        # When there is no power value in our data (which happens mainly when we
        # explicitely make tests without power data), we make up one with NaN values.
        if "power" not in _data:
            shape = tuple(_data.dims.values())
            _data["power"] = xr.DataArray(np.empty(shape) * np.nan, dims=tuple(_data.dims))

        data = _data["power"]

        # PV data dropout.
        if (
            # Dropout makes sense only during training.
            is_training
            and self._pv_dropout > 0
            # This one is for mypy.
            and self._random_state is not None
            and self._random_state.random() < self._pv_dropout
        ):
            data *= np.nan

        coords = _data.coords

        lat = float(coords["latitude"].values)
        lon = float(coords["longitude"].values)

        # Get the metadata from the PV data.
        tilt = self._tilt_getter(_data)
        orientation = self._orientation_getter(_data)
        capacity = self._capacity_getter(_data)

        # Drop every coordinate except `ts`.
        extra_var = set(data.coords).difference(["ts"])
        data = data.drop_vars(extra_var)

        # As usual we normalize the PV data wrt irradiance and our PV "factor".
        # Using `safe_div` with `np.nan` fallback to get `nan`s instead of `inf`. The `nan` are
        # ignored in `compute_history_per_horizon`.
        if self._normalize_features:
            # Get the theoretical irradiance for all the timestamps in our history.
            irr1 = get_irradiance(
                lat=lat,
                lon=lon,
                timestamps=data.coords["ts"],
                tilt=tilt,
                orientation=orientation,
            )
            norm_data = safe_div(data, irr1["poa_global"].to_numpy() * capacity, fallback=np.nan)
        else:
            norm_data = data

        history = compute_history_per_horizon(
            norm_data,
            now=x.ts,
            horizons=self.config.horizons,
        )

        # Get the middle timestamp for each of our horizons.
        horizon_timestamps = [
            x.ts + timedelta(minutes=(f0 + f1) / 2) for f0, f1 in self.config.horizons
        ]

        recent_power_minutes = 30

        # Get the irradiance for those timestamps.
        irr2 = get_irradiance(
            lat=lat,
            lon=lon,
            # We add a timestamp for the recent power, that we'll treat separately afterwards.
            timestamps=horizon_timestamps + [x.ts - timedelta(minutes=recent_power_minutes / 2)],
            tilt=tilt,
            orientation=orientation,
        )

        # TODO Should we use the other values from `get_irradiance` other than poa_global?
        poa_global: np.ndarray = irr2.loc[:, "poa_global"].to_numpy()

        poa_global_now = poa_global[-1]
        poa_global = poa_global[:-1]

        # Features that start with '_' are not sent to the regressor but are used for
        # things like normalization. See the regressor code.
        features["_poa_global"] = poa_global
        features["_capacity"] = self._vectorize_feature(capacity)

        # In the case of poa_global, we also want to use as features.
        features["poa_global"] = poa_global

        # In our current setup, features must be 1D numpy arrays (one value per
        # horizon). However we do have some values that are the same for each horizon.
        # We note them in this dictionary and we'll vectorize them later.
        scalar_features: dict[str, float] = {}

        if self._version >= 2 and self._use_capacity_as_feature:
            scalar_features["capacity"] = capacity if np.isfinite(capacity) else -1.0

        for agg in ["max", "mean", "median"]:
            # When the array is empty or all nan, numpy emits a warning. We don't care about those
            # and are happy with np.nan as a result.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                warnings.filterwarnings("ignore", r"Mean of empty slice")
                # Use the `nan` version of the aggregators.
                aggregated = getattr(np, "nan" + agg)(history, axis=1)

            assert len(aggregated) == len(self.config.horizons)
            features["h_" + agg + "_nan"] = np.isnan(aggregated) * 1.0
            features["h_" + agg] = np.nan_to_num(aggregated)

        if self._nwp_data_sources is not None:
            for source_key, source in self._nwp_data_sources.items():
                if source._tolerance is not None:
                    tolerance = str(source._tolerance)
                else:
                    tolerance = None

                if (
                    is_training
                    and self._nwp_dropout > 0.0
                    and self._random_state is not None
                    and self._random_state.random() < self._nwp_dropout
                ):
                    nwp_data_per_horizon = None
                else:
                    nwp_data_per_horizon = source.get(
                        now=x.ts,
                        timestamps=horizon_timestamps,
                        nearest_lat=lat,
                        nearest_lon=lon,
                        tolerance=tolerance,
                    )

                nwp_variables = source.list_variables()

                for variable in nwp_variables:
                    # Deal with the trivial case where the returns NWP is simply `None`.
                    # This happens if there wasn't any data for the given tolerance.
                    if nwp_data_per_horizon is None:
                        var_per_horizon = np.array([np.nan for _ in self.config.horizons])
                    else:
                        var_per_horizon = nwp_data_per_horizon.sel(variable=variable).values

                    # Deal with potential NaN values in NWP.
                    var_per_horizon_is_nan = np.isnan(var_per_horizon) * 1.0
                    var_per_horizon = np.nan_to_num(
                        var_per_horizon, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    # We only want to append the name of the NWP variable to include the provider
                    # if there are multiple NWP data sources

                    if len(self._nwp_data_sources) > 1:
                        variable_source_key = variable + source_key

                    else:
                        variable_source_key = variable

                    features[variable_source_key] = var_per_horizon
                    features[variable_source_key + "_isnan"] = var_per_horizon_is_nan

        # add another section here fore getting the satellite data
        if self._satellite_data_sources is not None:
            # add the forecast horizon to the features. This is because the satellite data is
            # only available for the current time step, but not as a forecast, compared to NWP
            # which are available at all timesteps
            # not each horizon is the start and end horizon
            feature_forecast_horizons = []
            for horizon in self.config.horizons:
                feature_forecast_horizons.append((horizon[0] + horizon[1]) / 2.0)
            features["forecast_horizons"] = np.array(feature_forecast_horizons)

            # loop over satellite sources
            for source_key, source in self._satellite_data_sources.items():
                if source._tolerance is not None:
                    tolerance = str(source._tolerance)
                else:
                    tolerance = None

                if (
                    is_training
                    and self._nwp_dropout > 0.0
                    and self._random_state is not None
                    and self._random_state.random() < self._satellite_dropout
                ):
                    satellite_data = None
                else:
                    if self._satellite_patch_size > 0:
                        satellite_data = source.get(
                            now=x.ts,
                            timestamps=horizon_timestamps,
                            min_lat=lat - self._satellite_patch_size / 2,
                            max_lat=lat + self._satellite_patch_size / 2,
                            min_lon=lon - self._satellite_patch_size / 2,
                            max_lon=lon + self._satellite_patch_size / 2,
                            nearest_lon=lon,
                            tolerance=tolerance,
                        )

                        # take mean over x and y
                        if satellite_data is not None:
                            satellite_data = satellite_data.mean(dim=["x", "y"])

                    else:
                        satellite_data = source.get(
                            now=x.ts,
                            timestamps=horizon_timestamps,
                            nearest_lat=lat,
                            nearest_lon=lon,
                            tolerance=tolerance,
                        )
                satellite_variables = source.list_variables()

                for variable in satellite_variables:
                    # Deal with the trivial case where the returns Satellite is simply `None`.
                    # This happens if there wasn't any data for the given tolerance.
                    if satellite_data is not None:
                        var = satellite_data.sel(variable=variable).values

                        # expand satellite data to all time steps
                        var_per_horizon = np.array([var for _ in self.config.horizons])

                    else:
                        var_per_horizon = np.array([np.nan for _ in self.config.horizons])

                    # Deal with potential NaN values in NWP.
                    var_per_horizon_is_nan = np.isnan(var_per_horizon) * 1.0
                    var_per_horizon = np.nan_to_num(
                        var_per_horizon, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    # We only want to append the name of the Satellite variable to include the
                    # provider
                    # if there are multiple Satellite data sources
                    if len(self._satellite_data_sources) > 1:
                        variable_source_key = variable + source_key

                    else:
                        variable_source_key = variable

                    features[variable_source_key] = var_per_horizon
                    features[variable_source_key + "_isnan"] = var_per_horizon_is_nan

        # Get the recent power.
        recent_power = float(
            data.sel(ts=slice(x.ts - timedelta(minutes=recent_power_minutes), x.ts)).mean()
        )
        recent_power_nan = np.isnan(recent_power)

        # Normalize it.
        if self._normalize_features:
            recent_power = safe_div(recent_power, poa_global_now * capacity)

        scalar_features["recent_power"] = 0.0 if recent_power_nan else recent_power
        scalar_features["recent_power_nan"] = recent_power_nan * 1.0

        # recent power values
        recent_power_values = data.sel(
            ts=slice(x.ts - timedelta(minutes=recent_power_minutes), x.ts)
        ).values

        # make sure recent power values is the right length
        if not hasattr(self, "_n_recent_power_values"):
            self._n_recent_power_values = 0
        if len(recent_power_values) < self._n_recent_power_values:
            recent_power_values = np.pad(
                recent_power_values,
                (0, self._n_recent_power_values - len(recent_power_values)),
                "constant",
                constant_values=np.nan,
            )
        elif len(recent_power_values) > self._n_recent_power_values:
            recent_power_values = recent_power_values[
                len(recent_power_values) - self._n_recent_power_values :
            ]

        if self._normalize_features:
            recent_power_values = safe_div(recent_power_values, poa_global_now * capacity)

        for i, value in enumerate(recent_power_values):
            scalar_features[f"recent_power_values_{i}"] = value
            scalar_features[f"recent_power_values_{i}_isnan"] = np.isnan(value) * 1.0

        if self._version >= 2:
            scalar_features["poa_global_now_is_zero"] = poa_global_now == 0.0

        return features | {
            key: self._vectorize_feature(value) for key, value in scalar_features.items()
        }

    def train(
        self, train_iter: Iterable[Batch], valid_iter: Iterable[Batch], batch_size: int
    ) -> None:
        self._regressor.train(train_iter, valid_iter, batch_size)

    def explain(self, x: X):
        """Return the internal regressor's explain."""
        features = self.get_features(x)
        explanation = self._regressor.explain(features)
        return explanation

    def _v7_get_capacity(self, dataset: xr.Dataset) -> float:
        """Get capacity as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["factor"].values)
        else:
            try:
                # If there is a `capacity` variable in our data, we use that.
                if self._use_data_capacity:
                    # Take the first value, assuming the capacity doesn't change that rapidly.
                    return dataset["capacity"].values[0]
                else:
                    # Otherwise use some heuristic as capacity.
                    return float(dataset["power"].quantile(0.99))
            except Exception as e:
                _log.warning("Error while calculating capacity")
                _log.exception(e)
                return np.nan

    def _v7_get_tilt(self, dataset: xr.Dataset) -> float:
        """Get tilt as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["tilt"].values)
        else:
            return 35

    def _v7_get_orientation(self, dataset: xr.Dataset) -> float:
        """Get orientation as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["orientation"].values)
        else:
            return 180

    def get_state(self):
        state = self.__dict__.copy()
        # Do not save the data sources. Those should be set when loading the model using the `setup`
        # function.
        # We can't put that in __getstate__ directly because we need it when the model is pickled
        # for multiprocessing.
        del state["_pv_data_source"]
        del state["_nwp_data_sources"]
        del state["_satellite_data_sources"]
        return state

    def set_state(self, state):
        if "_version" not in state:
            raise RuntimeError("You are trying to load a deprecated model")

        if state["_version"] > _VERSION:
            raise RuntimeError(
                "You are trying to load a newer model in an older version of the code"
                f" ({state['_version']} > {_VERSION})."
            )

        # Load default arguments from older versions.
        if state["_version"] < 2:
            state["_use_inferred_meta"] = True
            state["_normalize_features"] = True
        if state["_version"] < 3:
            state["_use_data_capacity"] = False
        if state["_version"] < 4:
            state["_use_capacity_as_feature"] = True
        if state["_version"] < 5:
            state["_num_days_history"] = 7
        if state["_version"] < 6:
            state["_nwp_tolerance"] = None
        if state["_version"] < 7:
            state["_nwp_dropout"] = 0.0
        if state["_version"] < 8:
            state["_capacity_getter"] = self._v7_get_capacity
            state["_tilt_getter"] = self._v7_get_tilt
            state["_orientation_getter"] = self._v7_get_orientation
            state["_pv_dropout"] = 0.0

        super().set_state(state)
