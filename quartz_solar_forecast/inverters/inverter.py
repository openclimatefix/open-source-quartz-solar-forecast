import abc
from typing import Optional

import pandas as pd


class AbstractInverter(abc.ABC):

    @abc.abstractmethod
    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        raise NotImplementedError
