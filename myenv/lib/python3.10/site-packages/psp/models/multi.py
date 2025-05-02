import datetime as dt

import numpy as np

from psp.models.base import PvSiteModel
from psp.typings import Features, X, Y


class MultiPvSiteModel(PvSiteModel):
    """Higher-order model that wrap a series of models trained at different intervals.

    When evaluating a given sample, we use the latest model that was trained before that sample.

    This wrapper class should not be used in production: use the child model directly instead.
    """

    def __init__(self, models: dict[dt.datetime, PvSiteModel]):
        self._models = models
        # Make sure the models are sorted by date.
        assert list(sorted(models)) == list(models)

    def predict_from_features(self, x: X, features: Features) -> Y:
        model = self._get_model_for_ts(x.ts)
        return model.predict_from_features(x, features)

    def get_features(self, x: X, is_training: bool = False) -> Features:
        model = self._get_model_for_ts(x.ts)
        return model.get_features(x, is_training=is_training)

    def get_features_without_pv(self, x: X, is_training: bool = False) -> Features:
        """
        Generate a set of features from the input data `x`, excluding those derived from
        photovoltaic (PV) power data.

        Args:
        ----
            x (X): The input data to generate features from.
            is_training (bool, optional): A flag indicating whether the function is being called
            during training. Defaults to False.

        Returns:
        -------
            features: A dictionary of features, with PV-derived features set to NaN and NaN
            labelling features set to 1.
        """
        features = self.get_features(x, is_training=is_training)
        # List of features derived from 'pv'
        pv_derived_features = ["recent_power", "h_max", "h_median", "h_mean"]
        nan_pv_derived_features = ["recent_power_nan", "h_max_nan", "h_median_nan", "h_mean_nan"]
        recent_power_values_features = [f for f in features if f.startswith("recent_power_values")]
        recent_power_values_nans = [
            f for f in features if (f.startswith("recent_power_values")) and ("isnan" in f)
        ]
        for feature in pv_derived_features + recent_power_values_features:
            if feature in features:
                # Set the value to NaN
                features[feature] = np.full_like(features[feature], np.nan)

        for feature in nan_pv_derived_features + recent_power_values_nans:
            if feature in features:
                # Set the value to 1
                features[feature] = np.full_like(features[feature], 1)

        return features

    def _get_model_for_ts(self, ts: dt.datetime) -> PvSiteModel:
        # Use the most recent model whose train date is *before* `x.ts`. This was the most recent
        # model at time `x.ts`.
        for date, model in reversed(self._models.items()):
            if ts > date:
                return model
        else:
            raise ValueError(f"Date {ts} is before all the models")

    def set_data_sources(self, *args, **kwargs):
        for model in self._models.values():
            model.set_data_sources(*args, **kwargs)

    def explain(self, x: X):
        model = self._get_model_for_ts(x.ts)
        return model.explain(x)

    def get_train_date(self, ts: dt.datetime) -> dt.datetime:
        for date in reversed(self._models):
            if ts > date:
                return date
        else:
            raise ValueError(f"Date {ts} is before all the models")

    @property
    def config(self):
        # Assume all the configs are the same and return the first one.
        return next(iter(self._models.values())).config
