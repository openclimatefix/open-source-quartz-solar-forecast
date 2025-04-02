"""Define the interface of a Regressor, which is used in models."""

import abc
from typing import Iterable

import numpy as np

from psp.typings import Batch, Features


class Regressor(abc.ABC):
    @abc.abstractmethod
    def train(self, train_iter: Iterable[Batch], valid_iter: Iterable[Batch], batch_size: int):
        pass

    @abc.abstractmethod
    def predict(self, features: Features) -> np.ndarray:
        pass

    def explain(self, features: Features):
        raise NotImplementedError
