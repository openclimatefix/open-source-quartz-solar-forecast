"""Simplify the pv dataset in particular for data exploration.

Given the original file, cleans it and saves it, along with sampled datasets.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

from psp.clients.uk_pv.data import C, filter_rows, trim_pv


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=pathlib.Path, help="input file")
    parser.add_argument("-m", "--meta", type=pathlib.Path, help="metadata.csv file")
    parser.add_argument("output", type=pathlib.Path, help="output directory")
    parser.add_argument(
        "--many-versions", action="store_true", help="also make smaller versions of the dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rnd_state = np.random.RandomState(1234)

    meta = pd.read_csv(args.meta)
    df = pd.read_parquet(args.input)

    df = trim_pv(df, meta)

    # Remove timezone info
    df[C.date] = df[C.date].dt.tz_localize(None)

    # re-index
    df = df.set_index([C.id, C.date])
    df = df.sort_index()

    dfs = {}

    dfs["all"] = df

    if args.many_versions:
        # Make a couple of sampled datasets.
        dfs["1M"] = df.sample(1_000_000, random_state=rnd_state)
        dfs["10k"] = dfs["1M"].sample(10_000, random_state=rnd_state)

        n = 100

        # Keep `n` systems and make samples of those too.
        ss_n = df.index.get_level_values(0).unique().tolist()[:n]

        dfs[f"{n}"] = df.loc[ss_n]

        dfs[f"{n}_1M"] = dfs[f"{n}"].sample(1_000_000, random_state=rnd_state)
        dfs[f"{n}_10k"] = dfs[f"{n}_1M"].sample(10_000, random_state=rnd_state)

        # Glasgow region.
        LON_RANGE = [-4.537402, -3.940503]
        LAT_RANGE = [55.722169, 56.000524]

        m = meta.copy()
        for col, (low, high) in zip([C.lat, C.lon], [LAT_RANGE, LON_RANGE]):
            m = filter_rows(m, (m[col] < high) & (m[col] > low), "filter on " + col)
        ss_glas = m[C.id].unique().tolist()

        dfs["glasgow"] = filter_rows(
            df,
            df.index.get_level_values(0).isin(ss_glas),  # type: ignore
            "glasgow",
        )
        dfs["glasgow_10k"] = dfs["glasgow"].sample(10_000, random_state=rnd_state)

    for key, df in dfs.items():
        df.to_parquet(args.output / f"5min_{key}.parquet")


if __name__ == "__main__":
    main()
