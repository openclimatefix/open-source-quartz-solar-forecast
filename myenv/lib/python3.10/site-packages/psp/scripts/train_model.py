"""Train a pv-site model

This script is used to train a model.
"""
import datetime as dt
import importlib
import logging
import shutil
from collections import defaultdict
from typing import TYPE_CHECKING

import click
import numpy as np
import tqdm

from psp.exp_configs.base import TrainConfigBase
from psp.metrics import mean_absolute_error
from psp.models.base import PvSiteModel
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    log_level_opt,
    num_workers_opt,
)
from psp.serialization import save_model
from psp.typings import Sample
from psp.utils.printing import pv_list_to_short_str

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_log = logging.getLogger(__name__)

SEED_TRAIN = 1234
SEED_VALID = 4321


def _count(x):
    """Count the number of non-nan/inf values."""
    return np.count_nonzero(np.isfinite(x))


def _err(x):
    """Calculate the error (95% confidence interval) on the mean of a list of points.

    We ignore the nan/inf values.
    """
    return 1.96 * np.nanstd(x) / np.sqrt(_count(x))


def _eval_model(model: PvSiteModel, dataloader: "DataLoader[Sample]") -> None:
    """Evaluate a `model` on samples from a `dataloader` and log the error."""
    horizon_buckets = 8 * 60
    errors_per_bucket = defaultdict(list)
    all_errors = []
    for sample in tqdm.tqdm(dataloader):
        pred = model.predict(sample.x)
        error = mean_absolute_error(sample.y, pred)
        for (start, end), err in zip(model.config.horizons, error):
            bucket = start // horizon_buckets
            errors_per_bucket[bucket].append(err)
            all_errors.append(err)

    for i, errors in errors_per_bucket.items():
        bucket_start = i * horizon_buckets // 60
        bucket_end = (i + 1) * horizon_buckets // 60
        mean_err = np.nanmean(errors)
        # Error on the error!
        err_err = _err(errors)
        _log.info(f"[{bucket_start:<2}, {bucket_end:<2}[ : {mean_err:.3f} ± {err_err:.3f}")
    mean_err = np.nanmean(all_errors)
    err_err = _err(all_errors)
    _log.info(f"Total: {mean_err:.3f} ± {err_err:.3f}")


@click.command()
@exp_root_opt
@exp_name_opt
@exp_config_opt
@num_workers_opt
@log_level_opt
@click.option("-b", "--batch-size", default=32, show_default=True)
@click.option(
    "--num-test-samples",
    default=100,
    show_default=True,
    help="Number of samples to use to test on train and valid. Use 0 to skip completely.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Erase the output directory if it already exists",
    default=False,
    show_default=True,
)
@click.option(
    "--no-infer",
    is_flag=True,
    help="If set to True, evaluates the error on the train/valid sets. Default is True.",
    default=False,
    show_default=True,
)
def main(
    exp_root,
    exp_name,
    exp_config_name,
    num_workers,
    batch_size: int,
    num_test_samples: int,
    log_level: str,
    force: bool,
    no_infer: bool,
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    # This fixes problems when loading files in parallel on GCP.
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # https://github.com/fsspec/gcsfs/issues/379
    if num_workers > 0:
        # Import `torch` only if needed because it is slow to import.
        import torch

        torch.multiprocessing.set_start_method("spawn")

    exp_config_module = importlib.import_module("." + exp_config_name, "psp.exp_configs")
    exp_config: TrainConfigBase = exp_config_module.ExpConfig()

    output_dir = exp_root / exp_name
    if not output_dir.exists() or force:
        output_dir.mkdir(exist_ok=True)
    else:
        raise RuntimeError(f'Output directory "{output_dir}" already exists')

    # Also copy the config into the experiment.
    shutil.copy(f"./psp/exp_configs/{exp_config_name}.py", output_dir / "config.py")

    # Load the model.
    model = exp_config.get_model(random_state=np.random.RandomState(2023))

    pv_data_source = exp_config.get_pv_data_source()

    # Dataset
    pv_splits = exp_config.make_pv_splits(pv_data_source)

    train_date_splits = exp_config.get_date_splits().train_date_splits

    _log.info(f"Train PVs: {pv_list_to_short_str(pv_splits.train)}")
    _log.info(f"Valid PVs: {pv_list_to_short_str(pv_splits.valid)}")

    for i, train_date_split in enumerate(train_date_splits):
        date = train_date_split.train_date

        start_ts = max(
            date - dt.timedelta(days=train_date_split.train_days),
            pv_data_source.min_ts(),
        )
        end_ts = date

        _log.info(f"Train time range: [{start_ts}, {end_ts}]")

        data_loader_kwargs = dict(
            data_source=pv_data_source,
            horizons=model.config.horizons,
            num_workers=num_workers,
            shuffle=True,
            start_ts=start_ts,
            end_ts=end_ts,
            step=train_date_split.step_minutes,
        )

        # Delay this slow import here (because of pytorch).
        from psp.training import make_data_loader

        train_data_loader = make_data_loader(
            **data_loader_kwargs,
            get_features=lambda x: model.get_features(x, is_training=True),
            batch_size=batch_size,
            pv_ids=pv_splits.train,
            random_state=np.random.RandomState(SEED_TRAIN),
        )

        limit = 128

        # Ensure that way we always have the same valid set, no matter the batch size (for this we
        # need to have only whole batches).
        assert limit % batch_size == 0

        valid_data_loader = make_data_loader(
            **data_loader_kwargs,
            get_features=lambda x: model.get_features(x, is_training=False),
            pv_ids=pv_splits.valid,
            batch_size=batch_size,
            random_state=np.random.RandomState(SEED_VALID),
            # We shuffle to get a good sample of data points.
            limit=limit,
        )

        model.train(train_data_loader, valid_data_loader, batch_size)

        path = output_dir / f"model_{i}.pkl"
        _log.info(f"Saving model trained on {end_ts} to {path}")
        save_model(model, path)

        # Print the error on the train/valid sets.
        if num_test_samples > 0 and not no_infer:
            _log.info("Error on the train set")
            train_data_loader2 = make_data_loader(
                **data_loader_kwargs,
                get_features=lambda x: model.get_features(x, is_training=True),
                batch_size=None,
                pv_ids=pv_splits.train,
                limit=num_test_samples,
                random_state=np.random.RandomState(SEED_TRAIN),
            )
            _eval_model(model, train_data_loader2)

            _log.info("Error on the valid set")
            valid_data_loader2 = make_data_loader(
                **data_loader_kwargs,
                get_features=lambda x: model.get_features(x, is_training=False),
                batch_size=None,
                pv_ids=pv_splits.valid,
                limit=num_test_samples,
                random_state=np.random.RandomState(SEED_VALID),
            )
            _eval_model(model, valid_data_loader2)


if __name__ == "__main__":
    main()
