import abc

import pandas as pd


class AbstractInverter(abc.ABC):
    """
    An abstract base class representing an inverter which can provide a snapshot of live data.
    """

    @abc.abstractmethod
    def get_data(self, ts: pd.Timestamp) -> pd.DataFrame | None:
        raise NotImplementedError
