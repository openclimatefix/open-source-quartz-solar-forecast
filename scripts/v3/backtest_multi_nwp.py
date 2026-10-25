"""
Backtest the v3 (multi_nwp) model on the sites in sites.csv with day-ahead weather.

Each daily 00:00 forecast uses the Previous Runs API: hours 0-24 take the run issued the day
before (24-48 h lead), hours 24-48 the run issued two days before (48-72 h lead). A live
forecast has fresher runs than that, so these scores are slightly pessimistic.

Two ways of keeping the scored data out of training:
- multi_nwp_cv: 5 site folds x 4 season folds; each block is predicted by a model that saw
  neither its sites nor its season.
- multi_nwp_strict: one model trained on the dev sites before 2024-10-01, predicting the
  held-out sites from 2024-10-01 (unseen sites and strictly later months).

Writes <out>/multi_nwp.parquet: one row per site, init and half-hour (period end, naive GMT).

Usage (see README.md):
    python scripts/v3/backtest_multi_nwp.py --uk-pv-dir <uk_pv> --cache-dir <dir> --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    FEATURES,
    FIRST_INIT,
    LAST_INIT,
    day_ahead_features,
    gradient_boosting,
    load_sites,
    load_truth,
)

SITE_FOLDS, SEASON_FOLDS = 5, 4
STRICT_SPLIT = pd.Timestamp("2024-10-01")
STEPS = 96  # half-hours in 48 h


def site_rows(site: pd.Series, truth: pd.Series, cache_dir: Path) -> pd.DataFrame:
    """Every (init, half-hour) of the backtest for one site, with the features of the run a
    forecast issued at that init would have used."""
    inits = pd.date_range(FIRST_INIT, LAST_INIT, freq="D")
    steps = np.arange(1, STEPS + 1)
    init = np.repeat(inits.to_numpy(), STEPS)
    timestamp = init + np.tile(steps * np.timedelta64(30, "m"), len(inits))
    horizon = np.tile((steps - 1) / 2, len(inits))
    rows = pd.DataFrame({"ss_id": site["ss_id"], "init": init, "timestamp": timestamp})
    rows["horizon_hours"] = horizon
    rows = rows.reindex(columns=[*rows.columns, *FEATURES])

    day1 = horizon < 24
    for lead_day, at_lead in ((1, day1), (2, ~day1)):
        features = day_ahead_features(site, lead_day, cache_dir)
        values = features.reindex(rows.loc[at_lead, "timestamp"])[FEATURES].to_numpy()
        rows.loc[at_lead, FEATURES] = values
    rows["target"] = truth.loc[site["ss_id"]].reindex(rows["timestamp"]).to_numpy() / site["kWp"]
    rows["kwp"] = site["kWp"]
    rows["split"] = site["split"]
    return rows


def cross_validated(rows: pd.DataFrame, trainable: np.ndarray) -> np.ndarray:
    site_ids = np.sort(rows["ss_id"].unique())
    site_fold = rows["ss_id"].map(dict(zip(site_ids, np.arange(len(site_ids)) % SITE_FOLDS)))
    season = ((rows["timestamp"].dt.month - 4) % 12) // (12 // SEASON_FOLDS)
    predictions = np.full(len(rows), np.nan)
    for s in range(SITE_FOLDS):
        for m in range(SEASON_FOLDS):
            test = ((site_fold == s) & (season == m)).to_numpy()
            train = rows[trainable & (site_fold != s).to_numpy() & (season != m).to_numpy()]
            model = gradient_boosting().fit(train[FEATURES], train["target"])
            predictions[test] = model.predict(rows.loc[test, FEATURES])
        print(f"[backtest_multi_nwp] CV site fold {s + 1}/{SITE_FOLDS}", flush=True)
    return predictions


def strict_forward(rows: pd.DataFrame, trainable: np.ndarray) -> np.ndarray:
    before = (rows["timestamp"] < STRICT_SPLIT).to_numpy()
    dev = (rows["split"] == "dev").to_numpy()
    train = rows[trainable & dev & before]
    target = ~dev & ~before
    predictions = np.full(len(rows), np.nan)
    model = gradient_boosting().fit(train[FEATURES], train["target"])
    predictions[target] = model.predict(rows.loc[target, FEATURES])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uk-pv-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sites = load_sites(args.uk_pv_dir)
    truth = load_truth(args.uk_pv_dir, sites)
    rows = pd.concat(
        [site_rows(site, truth, args.cache_dir) for _, site in sites.iterrows()],
        ignore_index=True,
    )
    # one training row per half-hour: the day-1 forecast, daytime only
    daytime = (rows["elevation"] > 0).to_numpy()
    trainable = (
        (rows["horizon_hours"] < 24).to_numpy() & daytime & rows["target"].notna().to_numpy()
    )

    kwp = rows["kwp"].to_numpy()
    rows["multi_nwp_cv"] = np.clip(cross_validated(rows, trainable), 0, None) * kwp
    rows["multi_nwp_strict"] = np.clip(strict_forward(rows, trainable), 0, None) * kwp

    args.out.mkdir(parents=True, exist_ok=True)
    columns = ["ss_id", "init", "timestamp", "multi_nwp_cv", "multi_nwp_strict"]
    rows[columns].to_parquet(args.out / "multi_nwp.parquet")
    print(f"[backtest_multi_nwp] wrote {args.out / 'multi_nwp.parquet'} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
