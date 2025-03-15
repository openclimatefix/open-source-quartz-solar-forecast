from typing import Callable

import numpy as np

from psp.typings import Y

# Base type of a metric.
# The output is an array because we predict for different timestamps in the future.
Metric = Callable[[Y, Y], np.ndarray]


def mean_absolute_error(y_true: Y, y_pred: Y) -> np.ndarray:
    assert y_true.powers.shape == y_pred.powers.shape
    return abs(y_true.powers - y_pred.powers)


class MeanRelativeError:
    def __init__(self, cap: float | None = None):
        self._cap = cap

    def __call__(self, y_true: Y, y_pred: Y) -> np.ndarray:
        err = abs(y_true.powers - y_pred.powers) / y_true.powers
        if self._cap is not None:
            return np.minimum(err, self._cap)
        else:
            return err
