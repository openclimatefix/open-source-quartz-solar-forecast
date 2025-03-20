from typing import TypeVar, overload

import numpy as np
import xarray as xr

T = TypeVar("T", bound=np.ndarray | xr.DataArray)


@overload
def safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    ...


@overload
def safe_div(num: T, den: T | float, fallback: float = 0.0) -> T:
    ...


@overload
def safe_div(num: T | float, den: T, fallback: float = 0.0) -> T:
    ...


def safe_div(num: T | float, den: T | float, fallback: float = 0.0) -> T | float:
    if isinstance(num, float) and isinstance(den, float):
        if den == 0 or not np.isfinite(den):
            return fallback
        return num / den

    return np.divide(  # type: ignore
        num, den, out=np.full_like(num, fallback), where=(den != 0) & np.isfinite(den)
    )


class MeanAggregator:
    """Utility class to track the mean of a value.

    Useful to track losses while training.
    """

    def __init__(self):
        self.reset()

    def add(self, value: float, n: int = 1):
        self._total += value
        self._n += n

    def mean(self):
        if self._n == 0:
            return 0.0
        return self._total / self._n

    def reset(self):
        self._total = 0.0
        self._n = 0
