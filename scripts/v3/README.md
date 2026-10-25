# v3 (multi_nwp): training and backtest

| File | What it does |
|---|---|
| `sites.csv` | The 49 uk_pv systems used, and which are `dev` / `held_out` |
| `common.py` | Loads sites and PV data, fetches and caches Open-Meteo, builds features |
| `backtest_gb.py` | Runs the shipped `gb` model (as on main, and with the #376 unit fixes) plus two naive baselines |
| `backtest_multi_nwp.py` | Runs `multi_nwp` with day-ahead weather, cross-validated and as a strict forward test |
| `score.py` | Scores everything on the held-out sites and writes `scores.md` |
| `train.py` | Trains the model shipped in `quartz_solar_forecast/models/v3-0.1.0.pkl` on all 49 sites |

## Data

PV data is the gated [uk_pv](https://huggingface.co/datasets/openclimatefix/uk_pv) dataset. Request access on Hugging Face, log in (`hf auth login`), then:

```bash
hf download openclimatefix/uk_pv --repo-type dataset --local-dir $UK_PV \
  --include "metadata.csv" --include "bad_data.csv" \
  --include "30_minutely/year=2024/*" --include "30_minutely/year=2025/*"
```

That's about 530 MB.

Weather comes from Open-Meteo (no key needed) and is cached as JSON under `$CACHE`:
- `previous_runs/`: ICON, ECMWF IFS and GFS from the Previous Runs API, i.e. the forecast for each hour from the run issued 1 and 2 days before. This is what `multi_nwp` trains and is tested on.
- `historical_forecast/`: ICON plus GFS visibility from the Historical Forecast API, for `gb`. Day-ahead cloud layers and visibility aren't archived, so `gb` gets this short-lead data instead. That's better weather than it gets live, so the comparison flatters `gb`.

The first run makes about 250 requests. Open-Meteo's free tier counts a year of hourly data as several calls, so this uses a few thousand of the 10,000 daily calls. If you hit the hourly limit the scripts wait and carry on.

## Running

From the repo root:

```bash
export UK_PV=/path/to/uk_pv CACHE=/path/to/cache OUT=/path/to/out

python scripts/v3/backtest_gb.py --uk-pv-dir $UK_PV --cache-dir $CACHE --out $OUT
python scripts/v3/backtest_multi_nwp.py --uk-pv-dir $UK_PV --cache-dir $CACHE --out $OUT
python scripts/v3/score.py --uk-pv-dir $UK_PV --out $OUT

# retrain the shipped model
python scripts/v3/train.py --uk-pv-dir $UK_PV --cache-dir $CACHE
```

On an M-series MacBook, with the weather already cached, `backtest_gb.py` takes about 30 minutes, `backtest_multi_nwp.py` 15, and the other two about a minute each.

## Setup

- **Sites:** 1-4 kWp, with tilt and orientation, reporting from March 2024 to April 2025, not listed in `bad_data.csv`. They're a random sample, filtered to systems reporting at least 95% as many midday readings as the most complete one. One system that reported zeros all year was dropped.
- **Split:** the sites were split at random into 24 `dev` and 25 `held_out`. Model choices were made on dev-site scores (I did see held-out scores for the first two simple baselines before setting this up). `score.py` reports held-out sites only.
- **Forecasts:** issued daily at 00:00 from 1 April 2024 to 30 March 2025, 48 hours ahead, scored per half-hour.
- **Metric:** mean absolute error over daytime half-hours (sun above the horizon), divided by capacity, averaged per site and then over sites. CIs come from resampling sites. Every forecast is scored on the same rows.
- **CV:** 5 site folds × 4 season folds. Each block is predicted by a model that saw neither those sites nor that season.
- **Strict forward:** one model trained on the dev sites up to 30 September 2024, scored on the held-out sites from 1 October 2024.
- **Day-ahead weather:** hours 0-24 use the run issued the day before (24-48 h lead) and hours 24-48 the run from two days before (48-72 h lead). A live forecast has newer runs than that.

## Expected output

```
| | CV, Apr 2024 - Mar 2025 | strict forward, Oct 2024 - Mar 2025 |
|---|---|---|
| gb (main) | 9.82 [9.29, 10.34] | 8.50 [7.83, 9.12] |
| gb + #376 units | 8.31 [7.85, 8.78] | 7.64 [7.03, 8.21] |
| yesterday's output | 11.60 [11.07, 12.13] | 10.11 [9.29, 10.89] |
| 7-day average | 10.02 [9.57, 10.48] | 8.99 [8.24, 9.71] |
| **multi_nwp** | 7.06 [6.65, 7.51] | 6.34 [5.80, 6.84] |

multi_nwp - gb (CV): -2.76 pp [-3.09, -2.43]
multi_nwp - gb (strict): -2.16 pp [-2.49, -1.83]
```

The strict numbers are lower for every model because they only cover October to March.
