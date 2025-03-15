import dataclasses
import datetime
from typing import Mapping

import numpy as np

PvId = str

Timestamp = datetime.datetime


@dataclasses.dataclass
class X:
    """Input for a PV site models."""

    pv_id: PvId
    # Time at which are making the prediction. Typically "now".
    ts: Timestamp


class Horizons:
    """The list of time horizons for which we want to make predictions.

    Each horizon can be seen as a `(start, end)` tuple where `start` and `end` are the minutes from
    "now" (the time at which we are making the prediction).

    This class behaves like a list of `(start, end)` integer tuples.
    """

    def __init__(self, duration: int, num_horizons: int):
        """
        Arguments:
        ---------
        duration: Duration (in minutes) of a single horizon.
        num_horizons: How many horizons to consider.
        """
        self._duration = duration
        self._num_horizons = num_horizons

    @property
    def duration(self):
        """Duration of a single horizon."""
        return self._duration

    def __len__(self):
        return self._num_horizons

    def __iter__(self):
        for i in range(self._num_horizons):
            yield self.__getitem__(i)

    def __getitem__(self, i):
        """Get the i-th horizon as a tuple [start, end] tuple in minutes."""
        if i < -len(self):
            raise IndexError
        if i >= self._num_horizons:
            raise IndexError

        if i < 0:
            i = len(self) - i

        return (self.duration * i, self.duration * (i + 1))


@dataclasses.dataclass
class Y:
    """Output for a PV site model."""

    # Power predictions for each horizon. `np.nan` means no prediction.
    powers: np.ndarray

    def __eq__(self, other) -> bool:
        return np.array_equal(self.powers, other.powers, equal_nan=True)


# At the moment we assume that each feature is a 1D array of shape `(num_horizons,)`.
# This could be generalized if we need features with other shapes.
Features = dict[str, np.ndarray]


@dataclasses.dataclass
class Sample:
    x: X
    y: Y
    features: Features


@dataclasses.dataclass
class BatchedX:
    pv_id: list[PvId]
    ts: list[Timestamp]


@dataclasses.dataclass
class BatchedY:
    powers: np.ndarray


BatchedFeatures = Mapping[str, np.ndarray]


@dataclasses.dataclass
class Batch:
    x: BatchedX
    # Note that `y.powers` here is a 2D array.
    y: BatchedY
    features: BatchedFeatures
