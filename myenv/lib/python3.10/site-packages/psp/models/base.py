"""Base classes for the PV site ml."""

import abc
import dataclasses
from typing import Any, Iterable

from psp.typings import Batch, Features, Horizons, X, Y


@dataclasses.dataclass
class PvSiteModelConfig:
    """Model meta data that all models must define."""

    horizons: Horizons


class PvSiteModel(abc.ABC):
    """Abstract interface for our models."""

    def __init__(self, config: PvSiteModelConfig):
        """Initialize.

        Don't forget to call `super().__init__(config)` in the children classes.
        """
        self._config = config

    @abc.abstractmethod
    def predict_from_features(self, x: X, features: Features) -> Y:
        """Predict the output from the features.

        Useful if the features were already computed, or to leverage
        computing features in parallel separately.
        """
        pass

    def predict(self, x: X) -> Y:
        """Predict the output from the input.

        This is what should be called in production.
        """
        features = self.get_features(x)
        return self.predict_from_features(x, features)

    @abc.abstractmethod
    def get_features(self, x: X, is_training: bool = False) -> Features:
        """Compute features for the model.

        This step will be run in parallel by our data pipelines.

        Arguments:
        ---------
            x: The input
            is_training: Indicate if we are training or not. This is useful for models
                that need a different behaviour in training VS testing, such as dropout.
        """
        pass

    # TODO Define the output type when we better understand what we need here.
    def explain(self, x: X):
        """Return some explanation of what the model does on input `x`."""
        raise NotImplementedError

    @property
    def config(self):
        return self._config

    def train(
        self, train_iter: Iterable[Batch], valid_iter: Iterable[Batch], batch_size: int
    ) -> None:
        """Train the model."""
        pass

    def set_data_sources(self, *args, **kwargs):
        """Set datasources, typically after having deserialized the model."""
        pass

    def get_state(self):
        """Return the necessary fields of the class for serialization.

        This is used by `psp.serialization` to save the model.

        We need a different hook than `__getstate__` because sometimes we want to customize the
        model serialization and the default pickling in different ways. An example of this is that
        we want pytorch's `DataLoader` to pickle our model alongside its data sources, but when we
        serialize a model, we don't want them.

        This is meant to be overridden in children classes if a custom behaviour is needed.
        """
        return self.__dict__.copy()

    def set_state(self, state: dict[str, Any]):
        """Set the state of `self` using fields from `state`.

        This is used by `psp.serialization` to load the model.

        This can be overriden in children classes.
        """
        self.__dict__.update(state)
