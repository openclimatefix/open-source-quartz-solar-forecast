"""
Train the v3 (multi_nwp) model on all sites in sites.csv and save it to the package.

Features come from day-1 Previous Runs forecasts (the run issued 24-48 h before each time), so
the model never trains on fresher weather than a live forecast has.

Usage (see README.md):
    python scripts/v3/train.py --uk-pv-dir <uk_pv> --cache-dir <dir> [--out <model.pkl>]
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from common import (
    DATA_END,
    FEATURES,
    FIRST_INIT,
    day_ahead_features,
    gradient_boosting,
    load_sites,
    load_truth,
)

from quartz_solar_forecast.forecasts.v3 import MODEL_PATH

TRAIN_START, TRAIN_END = FIRST_INIT, DATA_END - pd.Timedelta(days=7)


def training_rows(site: pd.Series, truth: pd.Series, cache_dir: Path) -> pd.DataFrame:
    features = day_ahead_features(site, lead_day=1, cache_dir=cache_dir)
    features["target"] = truth.loc[site["ss_id"]].reindex(features.index) / site["kWp"]
    in_window = (features.index > TRAIN_START) & (features.index <= TRAIN_END)
    return features[in_window & (features["elevation"] > 0) & features["target"].notna()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uk-pv-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(MODEL_PATH))
    args = parser.parse_args()

    sites = load_sites(args.uk_pv_dir)
    truth = load_truth(args.uk_pv_dir, sites)
    rows = pd.concat([training_rows(site, truth, args.cache_dir) for _, site in sites.iterrows()])
    print(f"[train] {len(rows):,} daytime half-hours from {len(sites)} sites")

    model = gradient_boosting().fit(rows[FEATURES], rows["target"])
    with open(args.out, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)
    print(f"[train] saved {args.out}")


if __name__ == "__main__":
    main()
