import datetime as dt
import functools
from operator import itemgetter
from typing import TYPE_CHECKING, Callable, Iterator, overload

import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterDataPipe

from psp.data_sources.pv import PvDataSource
from psp.typings import Batch, Features, Horizons, PvId, Sample, Timestamp, X, Y
from psp.utils.batches import batch_samples

if TYPE_CHECKING:
    # torch imports are slow so we only import if we need to.
    from torch.utils.data import DataLoader


def round_to(x, to=1):
    return round(x / to) * to


class PvXDataPipe(IterDataPipe[X]):
    """IterDataPipe that yields model inputs."""

    def __init__(
        self,
        data_source: PvDataSource,
        horizons: Horizons,
        pv_ids: list[PvId] | None = None,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
        step: int = 15,
        dataset_file: str | None = None,
    ):
        """
        Arguments:
        ---------
            pv_ids: If provided, will only pick from those pv_ids.
            start_ts: If provided, wll only pick dates after this date.
            end_ts: If provided, wll only pick dates before this date.
            step: Step used to make samples in time (in minutes).
            dataset_file: If provided, will only pick from those pv_ids.
                This has columns of pv_id and timestamp
        """
        self._data_source = data_source
        self._horizons = horizons
        self._pv_ids = pv_ids or self._data_source.list_pv_ids()
        self._start_ts = start_ts or self._data_source.min_ts()
        self._end_ts = end_ts or self._data_source.max_ts()
        self._step = step
        self._dataset_file = dataset_file

        # Sanity checks.
        assert len(self._pv_ids) > 0
        assert self._end_ts > self._start_ts

        if self._dataset_file is not None:
            self._dataset = pd.read_csv(self._dataset_file)

    def __iter__(self) -> Iterator[X]:
        step = dt.timedelta(minutes=self._step)

        if self._dataset_file is None:
            for pv_id in self._pv_ids:
                ts = self._start_ts
                minute = ts.minute
                ts = ts.replace(minute=round_to(minute, self._step), second=0, microsecond=0)
                while ts < self._end_ts:
                    x = X(pv_id=pv_id, ts=ts)
                    yield x
                    ts = ts + step
        else:
            for index, row in self._dataset.iterrows():
                pv_id = str(row["pv_id"])
                ts = pd.Timestamp(row["timestamp"])
                ts = ts.replace(second=0, microsecond=0)
                x = X(pv_id=pv_id, ts=ts)
                yield x


# We inherit from PvSamplesGenerator to save some code even though it's not super sound.
class RandomPvXDataPipe(PvXDataPipe):
    """Infinite loop iterator of random PV data points."""

    def __init__(
        self,
        data_source: PvDataSource,
        horizons: Horizons,
        random_state: np.random.RandomState,
        pv_ids: list[PvId] | None = None,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
        step: int = 1,
    ):
        """
        Arguments:
        ---------
            step: Round the timestamp to this many minutes (with 0 seconds and 0 microseconds).
        """
        self._random_state = random_state
        super().__init__(data_source, horizons, pv_ids, start_ts, end_ts, step)

    def __iter__(self) -> Iterator[X]:
        num_seconds = (self._end_ts - self._start_ts).total_seconds()

        while True:
            # Random PV.
            pv_id = self._random_state.choice(self._pv_ids)

            # Random timestamp
            delta_seconds = self._random_state.random() * num_seconds
            ts = self._start_ts + dt.timedelta(seconds=delta_seconds)

            # Round the minutes to a multiple of `steps`. This is particularly useful when testing,
            # where we might not want something as granualar as every minute, but want to be able
            # to aggregate many values for the *same* hour of day.
            minute = round_to(ts.minute, self._step)
            if minute > 59:
                minute = 0

            ts = ts.replace(minute=minute, second=0, microsecond=0)

            yield X(pv_id=pv_id, ts=ts)


def get_y_from_x(x: X, *, horizons: Horizons, data_source: PvDataSource) -> Y | None:
    """Given an input, compute the output.

    Return `None` if there is not output - it's simpler to filter those later.
    """
    min_horizon = min(i[0] for i in horizons)
    max_horizon = max(i[1] for i in horizons)
    data = data_source.get(
        x.pv_id,
        x.ts + dt.timedelta(minutes=min_horizon),
        x.ts + dt.timedelta(minutes=max_horizon),
    )["power"]

    if data.size == 0:
        return None

    # Find the targets for that pv/ts.
    # TODO Find a way to vectorize this.
    powers = []
    for start, end in horizons:
        ts0 = pd.Timestamp(x.ts + dt.timedelta(minutes=start))
        ts1 = pd.Timestamp(x.ts + dt.timedelta(minutes=end)) - dt.timedelta(seconds=1)

        power_values = data.sel(ts=slice(ts0, ts1))

        if power_values.size == 0:
            powers.append(np.nan)
        else:
            power = float(power_values.mean())
            powers.append(power)

    powers_arr = np.array(powers)

    if np.all(np.isnan(powers_arr)):
        return None

    return Y(powers=powers_arr)


def _is_not_none(x):
    return x is not None


def _build_sample(
    x: X,
    *,
    horizons: Horizons,
    data_source: PvDataSource,
    get_features: Callable[[X], Features],
) -> Sample | None:
    y = get_y_from_x(x, horizons=horizons, data_source=data_source)

    # Skip the heavy computation if the target doesn't make sense.
    if y is None:
        return None

    features = get_features(x)

    return Sample(x=x, y=y, features=features)


# We need to `overload` to have a different return type depending on if we batch or not.
@overload
def make_data_loader(
    *,
    data_source: PvDataSource,
    horizons: Horizons,
    pv_ids: list[PvId],
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    get_features: Callable[[X], Features],
    random_state: np.random.RandomState | None = None,
    batch_size: None = None,
    num_workers: int = 0,
    shuffle: bool = False,
    step: int = 1,
    limit: int | None = None,
    dataset_file: str | None = None,
) -> "DataLoader[Sample]":
    ...


@overload
def make_data_loader(
    *,
    data_source: PvDataSource,
    horizons: Horizons,
    pv_ids: list[PvId],
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    get_features: Callable[[X], Features],
    random_state: np.random.RandomState | None = None,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    step: int = 1,
    limit: int | None = None,
    dataset_file: str | None = None,
) -> "DataLoader[Batch]":
    ...


def make_data_loader(
    *,
    data_source: PvDataSource,
    horizons: Horizons,
    pv_ids: list[PvId],
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    get_features: Callable[[X], Features],
    random_state: np.random.RandomState | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    shuffle: bool = False,
    step: int = 1,
    limit: int | None = None,
    dataset_file: str | None = None,
) -> "DataLoader[Sample] | DataLoader[Batch]":
    """
    Arguments:
    ---------
        batch_size: Batch size. None means no batching.
        step: Step in minutes for the timestamps.
        limit: return only this number of samples.
        dataset_file: is a csv file that has the column pv_id and timestamp.
            These are used to generate the dataset.
    """
    pvx_datapipe: PvXDataPipe
    if shuffle:
        assert random_state is not None
        pvx_datapipe = RandomPvXDataPipe(
            data_source=data_source,
            horizons=horizons,
            random_state=random_state,
            pv_ids=pv_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            step=step,
        )
    else:
        pvx_datapipe = PvXDataPipe(
            data_source=data_source,
            horizons=horizons,
            pv_ids=pv_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            step=step,
            dataset_file=dataset_file,
        )

    # This has to be as early as possible to be efficient!
    pvx_datapipe = pvx_datapipe.sharding_filter()

    # This is the expensive part, where we compute our model-specific feature extraction.
    datapipe = pvx_datapipe.map(
        functools.partial(
            _build_sample,
            horizons=horizons,
            data_source=data_source,
            get_features=get_features,
        )
    )

    # `_build_sample` will return `None` when the sample is not useful (for instance when all the
    # targets have no data).
    datapipe = datapipe.filter(_is_not_none)

    # We add the ability to stop the pipeline after a `limit` number of samples.
    if limit is not None:
        datapipe = datapipe.header(limit if num_workers == 0 else np.ceil(limit / num_workers))

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size, wrapper_class=batch_samples)

    # Torch is slow to import so wait until we are sure we need it.
    from torch.utils.data import DataLoader

    data_loader: DataLoader[Sample] = DataLoader(
        datapipe,
        num_workers=num_workers,
        # We deal with the batches ourselves.
        batch_size=1,
        collate_fn=itemgetter(0),
    )

    return data_loader
