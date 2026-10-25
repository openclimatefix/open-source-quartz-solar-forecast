"""
Score the backtests: daytime nMAE (mean |forecast - actual| / capacity) per site, averaged
over sites, with 95% CIs from resampling sites (errors within a site are correlated, sites
are close to independent).

Every forecast is scored on the same rows: daytime half-hours (sun above the horizon at the
middle of the half-hour) where the actual and every forecast exist. Only held-out sites are
reported; the dev sites were used while developing the model.

Usage (see README.md):
    python scripts/v3/score.py --uk-pv-dir <uk_pv> --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtest_multi_nwp import STRICT_SPLIT
from common import load_sites, load_truth
from pvlib.solarposition import get_solarposition

BASELINES = ["gb", "gb_pr376", "persistence", "climatology"]
LABELS = {
    "gb": "gb (main)",
    "gb_pr376": "gb + #376 units",
    "persistence": "yesterday's output",
    "climatology": "7-day average",
    "multi_nwp": "**multi_nwp**",
}
BOOTSTRAP_SAMPLES = 2000


def load_backtests(uk_pv_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = load_sites(uk_pv_dir)
    keys = ["ss_id", "init", "timestamp"]
    rows = pd.read_parquet(out / "gb.parquet").merge(
        pd.read_parquet(out / "multi_nwp.parquet"), on=keys, how="inner", validate="1:1"
    )
    truth = load_truth(uk_pv_dir, sites)
    rows["actual"] = truth.reindex(
        pd.MultiIndex.from_frame(rows[["ss_id", "timestamp"]])
    ).to_numpy()
    rows = rows.merge(sites[["ss_id", "split", "kWp", "latitude_rounded", "longitude_rounded"]])
    rows["daytime"] = False
    for site in sites.itertuples():
        mask = (rows["ss_id"] == site.ss_id).to_numpy()
        midpoints = pd.DatetimeIndex(rows.loc[mask, "timestamp"]) - pd.Timedelta("15min")
        solar = get_solarposition(midpoints, site.latitude_rounded, site.longitude_rounded)
        rows.loc[mask, "daytime"] = solar["apparent_elevation"].to_numpy() > 0
    return rows, sites


def per_site_nmae(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    errors = rows[columns].sub(rows["actual"], axis=0).abs().div(rows["kWp"], axis=0)
    return errors.groupby(rows["ss_id"]).mean()


def bootstrap(per_site: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    values = per_site.to_numpy()
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return pd.DataFrame(
        {
            "nmae": values.mean(axis=0),
            "low": np.percentile(means, 2.5, axis=0),
            "high": np.percentile(means, 97.5, axis=0),
        },
        index=per_site.columns,
    )


def score(rows: pd.DataFrame, model_column: str) -> tuple[pd.DataFrame, pd.Series]:
    columns = BASELINES + [model_column]
    scored = rows[rows["daytime"] & rows["actual"].notna() & rows[columns].notna().all(axis=1)]
    per_site = per_site_nmae(scored, columns).rename(columns={model_column: "multi_nwp"})
    paired = bootstrap(per_site[["multi_nwp"]].sub(per_site["gb"], axis=0)).loc["multi_nwp"]
    return 100 * bootstrap(per_site), 100 * paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uk-pv-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, sites = load_backtests(args.uk_pv_dir, args.out)
    held_out = rows[rows["split"] == "held_out"]
    cv, cv_paired = score(held_out, "multi_nwp_cv")
    strict, strict_paired = score(
        held_out[held_out["timestamp"] >= STRICT_SPLIT], "multi_nwp_strict"
    )

    n_sites = (sites["split"] == "held_out").sum()
    lines = [
        f"Daytime nMAE (% of capacity), 0-48 h, {n_sites} held-out sites, 95% CI",
        "",
        "| | CV, Apr 2024 - Mar 2025 | strict forward, Oct 2024 - Mar 2025 |",
        "|---|---|---|",
    ]
    for name, label in LABELS.items():
        cells = [
            f"{t.loc[name, 'nmae']:.2f} [{t.loc[name, 'low']:.2f}, {t.loc[name, 'high']:.2f}]"
            for t in (cv, strict)
        ]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.append("")
    for label, p in (("CV", cv_paired), ("strict", strict_paired)):
        diff = f"{p['nmae']:+.2f} pp [{p['low']:+.2f}, {p['high']:+.2f}]"
        lines.append(f"multi_nwp - gb ({label}): {diff}")
    text = "\n".join(lines)
    (args.out / "scores.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
