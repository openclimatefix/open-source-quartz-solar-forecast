import abc
import datetime
import logging
import pathlib
from typing import TypeVar

import xarray as xr

from psp.typings import PvId, Timestamp
from psp.utils.dates import to_pydatetime

_ID = "pv_id"
_TS = "ts"

# https://peps.python.org/pep-0673/
_Self = TypeVar("_Self", bound="PvDataSource")


_log = logging.getLogger(__name__)


class PvDataSource(abc.ABC):
    """Definition of the interface for loading PV data."""

    @abc.abstractmethod
    def get(
        self,
        pv_ids: list[PvId] | PvId,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
    ) -> xr.Dataset:
        """Get a slice of the data as a xarray Dataset.

        The returned Dataset should have dimensions "pv_id" and "ts", and any `coords`.
        """
        pass

    @abc.abstractmethod
    def list_pv_ids(self) -> list[PvId]:
        pass

    @abc.abstractmethod
    def min_ts(self) -> Timestamp:
        pass

    @abc.abstractmethod
    def max_ts(self) -> Timestamp:
        pass

    @abc.abstractmethod
    def as_available_at(self: _Self, ts: Timestamp) -> _Self:
        """Return a copy of the data source that will filter anything that was not available at
        `ts`.

        This is a intended as a safety mechanism when we want to make sure we can't use data after
        a certain point in time. In particular, we don't want to be able to use data from the
        future when training models.

        Note that in general this does not necessarily means filtering everything before `ts`.
        Sometimes even less data is available at time `ts` (see for instance how we use the
        `lag_minutes` parameter in `NetcdfPvDataSource`).

        Arguments:
        ---------
            ts: The "now" timestamp, everything after is the future.
        """
        pass

    def list_data_variables(self) -> list[str]:
        raise NotImplementedError


def min_timestamp(a: Timestamp | None, b: Timestamp | None) -> Timestamp | None:
    """Util function to calculate the minimum between two timestamps that supports `None`.

    `None` values are assumed to be greater always.
    """
    if a is None:
        if b is None:
            return None
        else:
            return b
    else:
        # a is not None
        if b is None:
            return a
        else:
            return min(a, b)


class NetcdfPvDataSource(PvDataSource):
    def __init__(
        self,
        path_or_data: pathlib.Path | str | xr.Dataset,
        timestamp_dim_name: str = _TS,
        id_dim_name: str = _ID,
        rename: dict[str, str] | None = None,
        ignore_pv_ids: list[str] | None = None,
        lag_minutes: float = 0.0,
    ):
        """
        Arguments:
        ---------
            filepath: File path of the netcdf file.
            timestamp_dim_name: Name for the timestamp dimensions in the dataset.
            id_dim_name: Name for the "id" dimensions in the dataset.
            rename: This is passed to `xarray` to rename any coordinates or variable.
            ignore_pv_ids: The PV ids from this list will be completely ignored.
            lag_minutes: This represents the time (in minutes) it takes before the data is available
                in practice. Concretely, this means that when we call `as_available_at`,
                `lag_minutes` minutes will subtracted from the passed timestamp. When training, this
                should be set to the expected delay before the PV data is available, in production.
        """
        if rename is None:
            rename = {}

        if isinstance(path_or_data, xr.Dataset):
            self._path = None
            raw_data = path_or_data
        else:
            self._path = path_or_data
            raw_data = xr.open_dataset(self._path)

        self._timestamp_dim_name = timestamp_dim_name
        self._id_dim_name = id_dim_name
        self._rename = rename
        self._ignore_pv_ids = ignore_pv_ids
        self._lag_minutes = lag_minutes

        self._prepare_data(raw_data)

        self._set_max_ts(None)

    def _set_max_ts(self, ts: Timestamp | None) -> None:
        # See `as_available_at`.
        self._max_ts = ts

    def _prepare_data(self, raw_dataset: xr.Dataset) -> None:
        # Xarray doesn't like trivial renamings so we build a mapping of what actually changes.
        rename_map: dict[str, str] = {}

        if self._id_dim_name != _ID:
            rename_map[self._id_dim_name] = _ID
        if self._timestamp_dim_name != _TS:
            rename_map[self._timestamp_dim_name] = _TS

        rename_map.update(self._rename)

        self._data = raw_dataset.rename(rename_map)

        # We use `str` types for ids throughout.
        self._data.coords[_ID] = self._data.coords[_ID].astype(str)

        if self._ignore_pv_ids is not None:
            num_pvs_before = len(self._data.coords["pv_id"])
            self._data = self._data.drop_sel(pv_id=self._ignore_pv_ids)
            num_pvs = len(self._data.coords["pv_id"])
            _log.debug(f"Removed {num_pvs_before - num_pvs} PVs")

    def get(
        self,
        pv_ids: list[PvId] | PvId,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
    ) -> xr.Dataset:
        end_ts = min_timestamp(self._max_ts, end_ts)
        return self._data.sel(pv_id=pv_ids, ts=slice(start_ts, end_ts))

    def list_pv_ids(self):
        out = list(self._data.coords[_ID].values)

        if len(out) > 0:
            assert isinstance(out[0], PvId)

        return out

    def min_ts(self):
        ts = to_pydatetime(self._data.coords[_TS].min().values)  # type:ignore
        return min_timestamp(ts, self._max_ts)

    def max_ts(self):
        ts = to_pydatetime(self._data.coords[_TS].max().values)  # type:ignore
        return min_timestamp(ts, self._max_ts)

    def as_available_at(self, ts: Timestamp) -> "NetcdfPvDataSource":
        now = ts - datetime.timedelta(minutes=self._lag_minutes) - datetime.timedelta(seconds=1)
        # We simply make a copy and change it's `max_ts`.
        # Using `copy.copy` used __setstate__ and __getstate__, which we tampered with so we use our
        # own implementation, which is inspired from the pickle doc:
        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        new_ds = NetcdfPvDataSource.__new__(NetcdfPvDataSource)
        new_ds.__dict__.update(self.__dict__)
        new_ds._set_max_ts(min_timestamp(self._max_ts, now))
        return new_ds

    def __getstate__(self):
        # Prevent pickling (potentially big) data sources when we don't have a path. Having a path
        # means we don't need to save the data itself.
        if self._path is None:
            raise RuntimeError(
                "You can only pickle `PvDataSource`s that were constructed using a path"
            )
        d = self.__dict__.copy()
        # I'm not sure of the state contained in a `Dataset` object, so I make sure we don't save
        # it.
        del d["_data"]
        return d

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        # Only data sources with a path should have been pickled.
        assert self._path is not None
        self._prepare_data(xr.open_dataset(self._path))

    def list_data_variables(self) -> list[str]:
        return list(self._data.data_vars)
