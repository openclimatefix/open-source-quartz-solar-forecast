"""Define a generic regressor that wraps a scikit-learn regressor."""

import logging
from itertools import islice
from typing import Any, Iterable

import numpy as np
import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor

from psp.models.regressors.base import Regressor
from psp.typings import Batch, BatchedFeatures, Features
from psp.utils.batches import batch_features, concat_batches

_log = logging.getLogger(__name__)

_VERSION = 1


class SklearnRegressor(Regressor):
    """Regressor that wraps any scikit-learn regressor.

    We first flatten the all the horizon and actually train one regressor for all the horizons.
    We also do the target normalization (using pvlib's poa_global) in here.
    """

    def __init__(
        self, num_train_samples: int, normalize_targets: bool = True, sklearn_regressor=None
    ):
        """
        Arguments:
        ---------
            num_samples: Number of samples to train the forest on.
        """
        # Using absolute error alleviates some outlier problems.
        # Squared loss (the default) makes the big outlier losses more important.
        if sklearn_regressor is None:
            sklearn_regressor = HistGradientBoostingRegressor(
                loss="absolute_error",
                random_state=1234,
            )
        self._regressor = sklearn_regressor

        self._num_train_samples = num_train_samples
        self._normalize_targets = normalize_targets

        self._version = _VERSION

    def _prepare_features(
        self, features: BatchedFeatures, *, is_training: bool
    ) -> tuple[np.ndarray, list[str]]:
        """Build a (sample, feature)-shaped matrix from (batched) features.

        Optionally also build a list of feature names that match the columns.

        Return:
        ------
            A numpy array (rows=sample, columns=features) and the name of the columns as a list of
            string.
        """
        # Ignore the features that start with an underscore.
        feature_names = list([n for n in features if not n.startswith("_")])

        if is_training:
            # At train-time, we note the feature names.
            # We make a copy because this list is modified later.
            self._feature_names = list(feature_names)
        else:
            # At evaluation-time, we make sure that the feature names are the same as at train-time.
            feature_set = set(feature_names)
            train_feature_set = set(self._feature_names)

            if feature_set != train_feature_set:
                raise RuntimeError(
                    f"regressor was trained on features {train_feature_set} != {feature_set}"
                )

        # Stack all the features together. Note that we stack them in the order they were seen at
        # train time.
        new_features = np.stack([features[n] for n in self._feature_names], axis=-1)

        n_batch, n_horizon, n_features = new_features.shape
        assert n_features == len(feature_names)

        # Add the horizon index as a feature.
        horizon_idx = np.broadcast_to(
            np.arange(n_horizon, dtype=float), (n_batch, n_horizon)
        ).reshape(n_batch, n_horizon, 1)

        # (batch * horizon, features + 1)
        new_features = np.concatenate([new_features, horizon_idx], axis=2)

        n_features += 1

        feature_names.append("horizon_idx")

        assert new_features.shape == (
            n_batch,
            n_horizon,
            n_features,
        )

        assert len(feature_names) == n_features

        # Finally we flatten the horizons.
        new_features = new_features.reshape(n_batch * n_horizon, n_features)

        return new_features, feature_names

    def train(
        self,
        train_iter: Iterable[Batch],
        valid_iter: Iterable[Batch],
        batch_size: int,
    ):
        num_samples = self._num_train_samples

        num_batches = num_samples // batch_size
        # We put `tqdm` here because that's the slow part that we can put a progress bar on.
        _log.info("Extracting the features.")
        batches = [b for b in tqdm.tqdm(islice(train_iter, num_batches), total=num_batches)]

        # Concatenate all the batches into one big batch.
        batch = concat_batches(batches)

        # Make it into a (sample, features)-shaped matrix.
        xs, _ = self._prepare_features(batch.features, is_training=True)

        # (batch, horizon)
        poa = batch.features["_poa_global"]
        assert len(poa.shape) == 2

        # (batch, horizon)
        ys = batch.y.powers

        capacity = batch.features["_capacity"]
        assert capacity.shape == poa.shape

        # We can ignore the division by zeros, we treat the nan/inf later.
        with np.errstate(divide="ignore"):
            # No safe div because we want to ignore the points where `poa == 0`, we will multiply by
            # zero anyway later, so no need to learn that. Otherwise the model will spend a lot of
            # its capacity trying to learn that nights mean zero. Even if in theory it sounds like
            # a trivial decision to make for the model (if we use "poa_global" as a feature), in
            # practice ignoring those points seems to help a lot.
            if self._normalize_targets:
                ys = ys / (poa * capacity)
            else:
                # Hack to make sure we have `nan` when poa=0 even if we don't normalize.
                ys = ys / (poa * capacity) * (poa * capacity)

        # Flatten the targets just like we "flattened" the features.
        ys = ys.reshape(-1)

        # Remove `nan`, `inf`, etc. from ys.
        mask = np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

        # In practice this does not improve the results, but it does improve the model explanation.
        sample_weight = poa.reshape(-1)[mask]

        _log.info("Fitting the forest.")
        self._regressor.fit(xs, ys, sample_weight=sample_weight)

    def predict(self, features: Features):
        new_features, _ = self._prepare_features(batch_features([features]), is_training=False)
        pred = self._regressor.predict(new_features)
        assert len(pred.shape) == 1

        if self._normalize_targets:
            return pred * features["_capacity"] * features["_poa_global"]
        else:
            # Return 0. when poa_global is 0., otherwise the value.
            return pred * ((features["poa_global"] > 0) * 1.0)

    def explain(self, features: Features) -> tuple[Any, list[str]]:
        """Return the `shap` values for our sample, alonside the names of the features.

        We return a `shap` object that contains as many values as we have horizons for the sample
        (since internally we split those in individual samples before sending to the model).
        """
        try:
            import shap
        except ImportError:
            print("You need to install `shap` to use the `explain` functionality")
            return None, []

        batch = batch_features([features])

        new_features, new_feature_names = self._prepare_features(batch, is_training=False)

        explainer = shap.Explainer(self._regressor, feature_names=new_feature_names)
        shap_values = explainer(new_features)
        # TODO we should return something that is not shap-specific
        return shap_values

    def __setstate__(self, state):
        # Backward compatibility for before we had the _version field.
        if "_version" not in state:
            state["_normalize_targets"] = True
            state["_regressor"] = state["_tree"]
        self.__dict__ = state


# For backward compatibility.
ForestRegressor = SklearnRegressor
