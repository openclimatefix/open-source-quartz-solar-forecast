"""Neural net regressor.

*** Warning: work in progress and not well maintained ***

Initially used in the `RecentHistoryModel`
but swapped for decision trees which were simpler to optimize.
"""

from typing import Iterable

import numpy as np
import torch
import tqdm
from torch import nn

from psp.models.regressors.base import Regressor
from psp.typings import Batch, BatchedFeatures, Features, Horizons
from psp.utils.maths import MeanAggregator, safe_div


class NN(nn.Module):
    def __init__(self, num_features: int, num_horizon: int):
        super().__init__()
        self._num_horizon = num_horizon

        self.stack = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, per_horizon, common):
        # per_horizon: (batch, horizon, f1)
        # common: (batch, f2)

        # (batch, horizon, f2)
        # TODO I know that `Tensor.expand` doesn't allocated new memory. Is it really what we want
        # here? I'm afraid it could be slower than simply copying the data using `Tensor.cat`
        common = common.unsqueeze(1).expand(-1, self._num_horizon, -1)

        # (batch, horizon, f1 + f2)
        features = torch.cat([per_horizon, common], 2)

        return self.stack(features).squeeze(-1)


class NNRegressor(Regressor):
    def __init__(self, num_features: int, horizons: Horizons):
        self._nn = NN(num_features, num_horizon=len(horizons))

    def _batch_to_tensors(self, features: BatchedFeatures, y: np.ndarray, device: str):
        # Start by normalizing the `y` with irradiance and factor.
        # (batch, ts_pred)
        irr = features["irradiance"]
        # (batch, )
        factor = features["factor"].reshape(-1, 1)

        y = safe_div(y, irr * factor)

        # We set the missing data to 0. If it was NaN because it was at night, then it's fine.
        # TODO  If it was nan because of missing data, then the model will try to make those 0.
        # This could introduce some problems. The solution would be to ignore those.
        y = np.nan_to_num(y)

        yy = torch.tensor(y, dtype=torch.float32, device=device)

        return features, yy

    def _forward_batch(self, features: dict[str, np.ndarray], device: str) -> torch.Tensor:
        # (batch, horizon, features)
        per_horizon = torch.tensor(features["per_horizon"], dtype=torch.float32, device=device)

        # (batch, features2)
        common = torch.tensor(features["common"], dtype=torch.float32, device=device)

        return self._nn(per_horizon, common)

    # TODO We should probably extract some of this code that could be share with future models.
    # TODO Inject some of the dependencies (optimizer, loss function, etc.)
    def train(
        self, train_iter: Iterable[Batch], valid_iter: Iterable[Batch], batch_size: int
    ) -> None:
        torch.manual_seed(123)
        optimizer = torch.optim.Adam(self._nn.parameters(), lr=2e-4, weight_decay=1e-5)
        loss_func = torch.nn.L1Loss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._nn.train()
        train_loss_agg = MeanAggregator()

        validate_every = 10000 // batch_size

        for i, batch in tqdm.tqdm(enumerate(train_iter), unit="smpl", unit_scale=batch_size):
            if (i < validate_every and (i % (validate_every // 10) == 0)) or (
                i % validate_every == 0
            ):
                valid_loss_agg = MeanAggregator()
                self._nn.eval()
                for valid_batch in valid_iter:
                    features, y = self._batch_to_tensors(
                        valid_batch.features, valid_batch.y.powers, device
                    )
                    pred = self._forward_batch(features, device=device)
                    loss = loss_func(pred, y)
                    valid_loss_agg.add(loss.item())

                print(f"train loss: {train_loss_agg.mean()}")
                print(f"valid loss: {valid_loss_agg.mean()}")
                train_loss_agg.reset()
                self._nn.train()

                # for param in self._nn.parameters():
                #     print(param.data)

                features, y = self._batch_to_tensors(batch.features, y.powers, device)

            pred = self._forward_batch(features, device=device)
            # Compute prediction error
            loss = loss_func(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_agg.add(loss.item())

    def predict(self, features: Features):
        per_horizon = torch.tensor(features["per_horizon"], dtype=torch.float32)
        per_horizon = per_horizon.unsqueeze(0)
        common = torch.tensor(features["common"], dtype=torch.float32).unsqueeze(0)
        self._nn.eval()
        pred = self._nn(per_horizon, common).squeeze(0)
        return pred.detach().numpy()
