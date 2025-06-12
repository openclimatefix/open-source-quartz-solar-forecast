import datetime as dt
import logging
import pathlib
import pickle
from typing import Optional

# This import registers a codec.
import ocf_blosc2  # noqa
import xarray as xr

from psp.data_sources.utils import _STEP, _TIME, _VALUE, _VARIABLE, _X, _Y, slice_on_lat_lon
from psp.gis import CoordinateTransformer
from psp.typings import Timestamp
from psp.utils.dates import to_pydatetime
from psp.utils.hashing import naive_hash

_log = logging.getLogger(__name__)


class NwpDataSource:
    """Wrapper around a zarr file containing the NWP data.

    We assume that the data is a 5D tensor, with the dimensions being (time, step, x, y, variable).
    """

    def __init__(
        self,
        paths_or_data: str | list[str] | xr.Dataset,
        *,
        coord_system: int = 4326,
        x_dim_name: str = _X,
        y_dim_name: str = _Y,
        time_dim_name: str = _TIME,
        step_dim_name: str = _STEP,
        variable_dim_name: str = _VARIABLE,
        value_name: str = _VALUE,
        x_is_ascending: bool = True,
        y_is_ascending: bool = True,
        cache_dir: str | None = None,
        lag_minutes: float = 0.0,
        tolerance: Optional[str] = None,
        variables: Optional[list[str]] = None,
        filter_on_step: Optional[bool] = True,
    ):
        """
        Arguments:
        ---------
        paths_or_data: Path to the .zarr data or list of paths to different .zarr data or
            xarray dataset directly.
        coord_system: Integer representing the coordinate system for the position dimensions. 4326
            for (latitude, longitude), 27700 for OSGB, etc.
        *_dim_name: The 5 names of thedimensions in the data at `path`.
        value_name: The name of the value in the dataset at `path`.
        cache_dir: If provided, the `at_get` function will cache its result in the directory. This
            is useful when always training and testing on the same dataset, as the loading of the
            NWP is one of the main bottlenecks. Use with caution: it will create a lot of files!
        lag_minutes: Delay (in minutes) before the data is available. This is to mimic the fact that
            in production, the data is often late. We will add a "lag_minutes" of `lag_minutes`
            minutes when calling the `at` method.
        x_is_ascending: Is the `x` coordinate in ascending order. If it's in descending order, set
            this to `False`.
        y_is_ascending: Is the `y` coordinate in ascending order. If it's in descending order, set
            this to `False`.
        nwp_tolerance: How old should the NWP predictions be before we start ignoring them.
            See `NwpDataSource.get`'s documentation for details..
        nwp_variables: Only use this subset of NWP variables. Defaults to using all.

        """
        if isinstance(paths_or_data, str):
            paths_or_data = [paths_or_data]

        if isinstance(paths_or_data, xr.Dataset):
            self._paths = None
            raw_data = paths_or_data
        else:
            self._paths = paths_or_data
            raw_data = self._open(paths_or_data)
        # We'll have to transform the lat/lon coordinates to the internal dataset's coordinate
        # system.
        self._coordinate_transformer = CoordinateTransformer(4326, coord_system)

        self._x_dim_name = x_dim_name
        self._y_dim_name = y_dim_name
        self._time_dim_name = time_dim_name
        self._step_dim_name = step_dim_name
        self._variable_dim_name = variable_dim_name
        self._value_name = value_name
        self._x_is_ascending = x_is_ascending
        self._y_is_ascending = y_is_ascending

        self._lag_minutes = lag_minutes

        self._tolerance = tolerance
        self._variables = variables

        self._data = self._prepare_data(raw_data)
        self.raw_data = raw_data

        self._cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(exist_ok=True)

        self._filter_on_step = filter_on_step

    def _open(self, paths: list[str]) -> xr.Dataset:
        _log.debug(f"Opening data {paths}")
        return xr.open_mfdataset(
            paths,
            engine="zarr",
        )

    def _prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        # Rename the dimensions.
        rename_map: dict[str, str] = {}
        for old, new in zip(
            [
                self._x_dim_name,
                self._y_dim_name,
                self._time_dim_name,
                self._step_dim_name,
                self._variable_dim_name,
                self._value_name,
            ],
            [_X, _Y, _TIME, _STEP, _VARIABLE, _VALUE],
        ):
            if old != new:
                rename_map[old] = new

        data = data.rename(rename_map)

        # Filter data to keep only the variables in self._nwp_variables if it's not None
        if self._variables is not None:
            data = data.sel(variable=self._variables)

        # Sort data in time
        data = data.sortby(_TIME)

        return data

    def list_variables(self) -> list[str]:
        return list(self._data.coords[_VARIABLE].values)

    def get(
        self,
        *,
        now: Timestamp,
        timestamps: list[Timestamp] | Timestamp,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        nearest_lat: float | None = None,
        nearest_lon: float | None = None,
        tolerance: str | None = None,
        load: bool = True,
    ) -> xr.DataArray | None:
        """Slice the original data in the `time` dimension, and optionally on the x,y
        coordinates.

        There are two ways of filtering on the lat/lon coordinates:
            * Using the max/min_lat/lon, in which case we'll return many lat/lons.
            * Using the nearest_lat/lon, in which case we'll return the closest point.

        Arguments:
        ---------
        now: Time at which we are doing the query: we will use the closest "time of
            prediction" *before* this.
        timestamps: List of timestamps for which we want the predictions. For each of those, we'll
            return the *closest* prediction.
        min_lat: Lower bound on latitude (in degrees).
        max_lat: Upper bound on latitude.
        min_lon: Lower bound on longitude.
        max_lon: Upper bound on longitude.
        nearest_lat: Keep the data for the closest latitude.
        nearest_lon: Keep the data for the closest longitude.
        tolerance: Tolerance on the `now` timestamp. If the data is older than the tolerance, we'll
            return `None` instead of a DataArray. This argument is passed directly to
            `xarray.Dataset.sel`, see `xarray`'s doc for details.
        load: Should we explicitely load the data (call `.load()` on the `xarray.Dataset`). Defaults
            to `True` as loading typically makes subsequent usage of the data faster.

        Return:
        ------
        A `xarray.DataArray` object, or `None`.
        """
        if isinstance(timestamps, Timestamp):
            timestamps = [timestamps]

        for t in timestamps:
            if t < now:
                raise ValueError(f'Timestamp "{t}" should be after now={now}')

        # Only cache for nearest_* because lat/lon ranges could be big.
        use_cache = self._cache_dir

        data = None

        # Try to load it from the cache
        if use_cache:
            assert self._cache_dir is not None
            hash_data = [
                now,
                nearest_lat,
                nearest_lon,
                min_lat,
                max_lat,
                min_lon,
                max_lon,
                self._paths,
                self._lag_minutes,
                tolerance,
                *timestamps,
            ]
            hashes = tuple([naive_hash(x) for x in hash_data])
            hash_ = str(hash(hashes))
            path = self._cache_dir / hash_
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)

        # If it was not loaded from the cache, we load it from the original dataset.
        if data is None:

            data = self._get(
                now=now,
                timestamps=timestamps,
                nearest_lat=nearest_lat,
                nearest_lon=nearest_lon,
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                tolerance=tolerance,
                load=load,
            )
            # If using the cache, save it for next time.
            if use_cache:
                with open(path, "wb") as f:
                    pickle.dump(data, f, protocol=-1)

        return data

    def _get(
        self,
        *,
        now: Timestamp,
        timestamps: list[Timestamp],
        min_lat: float | None,
        max_lat: float | None,
        min_lon: float | None,
        max_lon: float | None,
        nearest_lat: float | None,
        nearest_lon: float | None,
        tolerance: str | None,
        load: bool,
    ) -> xr.DataArray | None:
        """Slice the data.

        This is the implementation of `.get` but without the caching mechanism.
        Use `.get` directly.
        """
        ds = self._data

        try:
            # Forward fill so that we get the value from the past, not the future!
            ds = ds.sel(
                {_TIME: now - dt.timedelta(minutes=self._lag_minutes)},
                method="ffill",
                tolerance=tolerance,  # type: ignore
            )
        except KeyError:
            # This happens when no data matches the tolerance.
            # Sanity check.
            assert tolerance is not None
            return None

        ds = slice_on_lat_lon(
            ds,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            nearest_lat=nearest_lat,
            nearest_lon=nearest_lon,
            transformer=self._coordinate_transformer,
            x_is_ascending=self._x_is_ascending,
            y_is_ascending=self._y_is_ascending,
        )

        init_time = to_pydatetime(ds[_TIME].values.item())

        # How long after `time` do we need the predictions.
        deltas = [t - init_time for t in timestamps]

        if self._filter_on_step:
            # Get the nearest prediction to what we are interested in.
            ds = ds.sel(step=deltas, method="nearest")

        da = ds[_VALUE]

        if load:
            da = da.load()

        return da

    def __getstate__(self):
        # Prevent pickling (potentially big) data sources when we don't have a path. Having a path
        # means we don't need to save the data itself.
        if self._paths is None:
            raise RuntimeError(
                "You can only pickle `NwpDataSource`s that were constructed using paths"
            )
        d = self.__dict__.copy()
        # I'm not sure of the state contained in a `Dataset` object, so I make sure we don't save
        # it.
        del d["_data"]
        return d

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        assert self._paths is not None
        self._data = self._prepare_data(self._open(self._paths))
