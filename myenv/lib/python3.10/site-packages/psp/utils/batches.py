"""Utils to deal with batches"""

import itertools
from typing import TypeVar

import numpy as np

from psp.typings import Batch, BatchedFeatures, BatchedX, BatchedY, Features, Sample

T = TypeVar("T")


def _chain_list(lists: list[list[T]]) -> list[T]:
    return list(itertools.chain.from_iterable(lists))


def concat_batched_features(batched_features: list[BatchedFeatures]) -> BatchedFeatures:
    assert len(batched_features) > 0
    keys = batched_features[0].keys()

    return {key: np.concatenate([b[key] for b in batched_features], axis=0) for key in keys}


def concat_batches(batches: list[Batch]) -> Batch:
    return Batch(
        x=BatchedX(
            pv_id=_chain_list([b.x.pv_id for b in batches]),
            ts=_chain_list([b.x.ts for b in batches]),
        ),
        y=BatchedY(
            powers=np.concatenate([b.y.powers for b in batches], axis=0),
        ),
        features=concat_batched_features([b.features for b in batches]),
    )


def batch_features(features: list[Features]) -> BatchedFeatures:
    keys = features[0].keys()
    return {key: np.stack([f[key] for f in features]) for key in keys}


def batch_samples(samples: list[Sample]) -> Batch:
    assert len(samples) > 0
    x = BatchedX(
        pv_id=[s.x.pv_id for s in samples],
        ts=[s.x.ts for s in samples],
    )
    y = BatchedY(powers=np.stack([s.y.powers for s in samples]))
    features = batch_features([s.features for s in samples])
    return Batch(x=x, y=y, features=features)
