"""Evaluate a model trained with the train_model.py script."""

import datetime as dt
import importlib
import logging
from typing import Optional

import click
import numpy as np
import pandas as pd
import tqdm

from psp.exp_configs.base import EvalConfigBase, TrainConfigBase, TrainEvalConfigBase
from psp.metrics import Metric, mean_absolute_error
from psp.models.multi import MultiPvSiteModel
from psp.scripts._options import exp_name_opt, exp_root_opt, log_level_opt, num_workers_opt
from psp.serialization import load_model
from psp.utils.interupting import continue_on_interupt
from psp.utils.printing import pv_list_to_short_str

METRICS: dict[str, Metric] = {
    # "mre_cap=1": MeanRelativeError(cap=1),
    "mae": mean_absolute_error,
}

_log = logging.getLogger(__name__)


@click.command()
@exp_root_opt
@exp_name_opt
@num_workers_opt
@log_level_opt
@click.option(
    "-l",
    "--limit",
    type=int,
    show_default=True,
    help="Maximum number of samples to consider."
    "Defaults to 1000 unless --sequential is passed, then defaults to no limit",
)
@click.option(
    "--eval-config",
    "eval_config_path",
    help='Use another config than the training one. e.g. "psp.exp_configs.some_config".',
)
@click.option(
    "--new-exp-name",
    help="Name of the experiment directory to save the results to."
    "Useful in combination with --eval-config.",
)
@click.option(
    "--split",
    "split_name",
    type=str,
    default="test",
    help="Split of the data to use: train | test.",
    show_default=True,
)
@click.option(
    "--test-start",
    type=click.DateTime(),
    help="Start date for the test set. Defaults to the start date specified in the config.",
)
@click.option(
    "--test-end",
    type=click.DateTime(),
    help="End date for the test set. Defaults to the end date specified in the config.",
)
@click.option(
    "--sequential",
    is_flag=True,
    help="By default we pick samples at random. Use this flag to run for all the timestamps."
    "The --step-minutes determines the frequency of the samples in time.",
)
@click.option(
    "--pv_ids",
    "pv_ids_one_string",
    type=str,
    help="Comma separated list of PV IDs to evaluate on. Defaults to all PVs in the test set.",
)
@click.option(
    "--step-minutes",
    type=int,
    default=None,
    help="The time interval of the samples in minutes. "
    "Defaults to None and then value is taken from the configuration.",
)
@click.option(
    "--test-dataset",
    type=str,
    default=None,
    help="The csv file for the test dataset to use. Defaults to None. "
    "The csv must have pv_id and timestamp columns.",
)
@click.option(
    "--no-live-pv",
    is_flag=True,
    default=False,
    help="Use no Live PV during inference to simulate no live PV in production.",
)
def main(
    exp_root,
    exp_name,
    num_workers,
    limit,
    split_name,
    log_level,
    eval_config_path,
    new_exp_name,
    test_start,
    test_end,
    sequential: bool,
    pv_ids_one_string: str,
    step_minutes: Optional[int] = None,
    test_dataset: Optional[str] = None,
    no_live_pv: bool = False,
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    if eval_config_path is not None:
        assert new_exp_name is not None

    assert split_name in ["train", "test"]

    # Add some default in non-random mode.
    if limit is None and not sequential:
        limit = 1000

    if num_workers > 0:
        # Only import torch here because it's a slow import.
        import torch

        # This fixes problems when loading files in parallel on GCP.
        # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
        # https://github.com/fsspec/gcsfs/issues/379
        torch.multiprocessing.set_start_method("spawn")

    print("Loading train config from ", f"{exp_root}.{exp_name}.config")
    exp_config_module = importlib.import_module(".config", f"{exp_root}.{exp_name}")
    train_config: TrainConfigBase = exp_config_module.ExpConfig()

    eval_config: EvalConfigBase
    if eval_config_path is not None:
        print("Loading eval config from ", eval_config_path)
        eval_config_module = importlib.import_module(eval_config_path)
        eval_config = eval_config_module.ExpConfig()
    else:
        print("Using the same config for eval.")
        assert isinstance(train_config, TrainEvalConfigBase)
        eval_config = train_config

    data_source_kwargs = eval_config.get_data_source_kwargs()
    pv_data_source = eval_config.get_pv_data_source()

    # Those are the dates we trained models for.
    date_splits = train_config.get_date_splits()
    # train_dates = dates_split.train_dates
    train_dates = [x.train_date for x in date_splits.train_date_splits]

    # Load the saved models.
    model_list = [
        load_model(exp_root / exp_name / f"model_{i}.pkl") for i in range(len(train_dates))
    ]
    models = {date: model for date, model in zip(train_dates, model_list)}
    # Wrap them into one big meta model.
    model = MultiPvSiteModel(models)

    model.set_data_sources(**data_source_kwargs)

    model_config = model.config

    # Setup the dataset.

    # TODO make sure the train_split from the model is consistent with the test one - we could
    # save in the model details about the training and check them here.
    pv_splits = eval_config.make_pv_splits(pv_data_source)
    if pv_ids_one_string is None:
        pv_ids = getattr(pv_splits, split_name)
    else:
        pv_ids = pv_ids_one_string.split(",")

    test_date_split = date_splits.test_date_split
    test_start = test_start or test_date_split.start_date
    test_end = test_end or test_date_split.end_date

    _log.info(f"Evaluating on PV split: {pv_list_to_short_str(pv_ids)}")
    _log.info(f"Time range: [{test_start}, {test_end}]")

    random_state = np.random.RandomState(1234)

    if step_minutes is None:
        step = test_date_split.step_minutes
    else:
        step = step_minutes

    # Delay this import because it itself imports pytorch which is slow.
    from psp.training import make_data_loader

    _log.info(f"No Live PV at Inference: {no_live_pv}")

    if no_live_pv:
        get_feature_function = model.get_features_without_pv
    else:
        get_feature_function = model.get_features

    data_loader = make_data_loader(
        data_source=pv_data_source,
        horizons=model_config.horizons,
        pv_ids=pv_ids,
        start_ts=test_start,
        end_ts=test_end,
        batch_size=None,
        random_state=random_state,
        get_features=get_feature_function,
        num_workers=num_workers,
        shuffle=(not sequential) and (test_dataset is None),
        step=step,
        limit=limit,
        dataset_file=test_dataset,
    )

    _log.info("Created data loader")
    print(data_loader)

    # Gather all errors for every samples. We'll make a DataFrame with it.
    error_rows = []

    pv_data_has_capacity = "capacity" in pv_data_source.list_data_variables()

    with continue_on_interupt(prompt=False):
        for i, sample in tqdm.tqdm(
            enumerate(data_loader),
            # Use rule of thumb when we don't have a `limit`.
            total=limit or ((test_end - test_start).total_seconds() / 60.0) // step * len(pv_ids),
        ):
            x = sample.x

            extra = {}

            if pv_data_has_capacity:
                capacity = pv_data_source.get(
                    pv_ids=x.pv_id, start_ts=x.ts - dt.timedelta(days=7), end_ts=x.ts
                )["capacity"].values[-1]
                extra["capacity"] = capacity

            y_true = sample.y
            y_pred = model.predict_from_features(x=x, features=sample.features)
            train_date = model.get_train_date(x.ts)
            for metric_name, metric in METRICS.items():
                error = metric(y_true, y_pred)
                # Error is a vector
                for i, (err_value, y, pred) in enumerate(zip(error, y_true.powers, y_pred.powers)):
                    horizon = model_config.horizons[i]
                    error_rows.append(
                        {
                            "pv_id": x.pv_id,
                            "ts": x.ts,
                            "ts_start": x.ts + dt.timedelta(minutes=horizon[0]),
                            "ts_end": x.ts + dt.timedelta(minutes=horizon[1]),
                            "metric": metric_name,
                            "error": err_value,
                            "horizon": horizon[0],
                            "y": y,
                            "pred": pred,
                            "train_date": train_date,
                            **extra,
                        }
                    )

    df = pd.DataFrame.from_records(error_rows)

    # print out the mae per horizon
    mae_per_horizon = {}
    for horizon in model_config.horizons:

        mae = df[df["horizon"] == horizon[0]]["error"].abs().mean()
        mae_per_horizon[horizon[0]] = mae
        print(f"MAE for horizon {horizon[0]}: {mae:.2f}")

    exp_name = exp_name or dt.datetime.now().isoformat()

    output_dir = exp_root / (new_exp_name or exp_name)
    print(f"Saving results to {output_dir}")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / f"{split_name}_errors.csv.gz", index=False)


if __name__ == "__main__":
    main()
