"""
We use python classes as configuration for our experiments.
Those are split in training and evaluation configs, but most of the time we make one big
config that satisfies both interfaces.
"""

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from psp.data_sources.pv import PvDataSource
    from psp.dataset import DateSplits
    from psp.models.base import PvSiteModel


class _ConfigBaseCommon(abc.ABC):
    @abc.abstractmethod
    def get_pv_data_source(self) -> "PvDataSource":
        """Get the PV data source used for the targets."""
        pass

    # TODO This comprises training and evaluation and should probably be split in two.
    @abc.abstractmethod
    def make_pv_splits(self, pv_data_source: "PvDataSource"):
        """Make the dataset splits from the pv data source."""
        pass

    # TODO This comprises training and evaluation and should probably be split in two.
    @abc.abstractmethod
    def get_date_splits(self) -> "DateSplits":
        pass


class TrainConfigBase(_ConfigBaseCommon, abc.ABC):
    """Defines the interface of an training config."""

    @abc.abstractmethod
    def get_model(self, *, random_state: np.random.RandomState | None = None) -> "PvSiteModel":
        """Get the model"""
        pass


class EvalConfigBase(_ConfigBaseCommon, abc.ABC):
    """Defines the interface of an evaluation config."""

    @abc.abstractmethod
    def get_data_source_kwargs(self) -> dict[str, Any]:
        """
        Get the keyword arguments that we pass to the `set_data_sources` method of the
        `PvSiteModel`.
        """
        pass


class TrainEvalConfigBase(TrainConfigBase, EvalConfigBase):
    """Experiment config that deals with both training and evaluation.

    This is the config that we typically use.
    """

    pass


# Backward compatibility
ExpConfigBase = TrainEvalConfigBase
