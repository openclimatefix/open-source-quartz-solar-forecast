import numpy as np
import pandas as pd

from quartz_solar_forecast.eval.metrics import bootstrap_mae_ci, metrics


def make_results(pv_ids, horizon_hour, rows_per_site, rng, error_scale=None, error=None):
    """Generation is daytime (>0.1) and forecast is generation plus a known
    error, so the MAE is known by construction. Errors are drawn PER ROW when
    error_scale is given — identical error sets across sites would make the
    site-bootstrap degenerate and the test would assert nothing."""
    rows = []
    for pv_id in pv_ids:
        for _ in range(rows_per_site):
            generation = 1.0 + rng.random()
            e = error if error is not None else rng.normal(0, error_scale)
            rows.append(
                {
                    "pv_id": pv_id,
                    "timestamp": "2024-01-01 12:00",
                    "horizon_hour": horizon_hour,
                    "forecast_power": generation + e,
                    "generation_power": generation,
                }
            )
    return pd.DataFrame(rows)


def make_metadata(df):
    return pd.DataFrame({"pv_id": df["pv_id"].unique(), "capacity": 2.0})


def test_mae_matches_known_errors_and_result_is_returned():
    rng = np.random.default_rng(0)
    # constant +0.2 error everywhere -> MAE is exactly 0.2 in every group
    results_df = make_results(pv_ids=range(5), horizon_hour=1, rows_per_site=8, rng=rng, error=0.2)
    out = metrics(results_df, make_metadata(results_df))

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {
        "horizon_group",
        "n",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "mae_normalized",
    }
    assert (out["mae"] == 0.2).all()
    # constant errors: every site resample gives the same MAE (up to float eps)
    assert np.allclose(out["mae_ci_low"], 0.2)
    assert np.allclose(out["mae_ci_high"], 0.2)


def test_ci_width_tracks_the_number_of_sites():
    # Regression test for the previous hardcoded n=50 SEM: a group with few
    # sites must report a wider interval than a group with many sites drawn
    # from the same error distribution.
    rng = np.random.default_rng(1)
    few = make_results(
        pv_ids=range(4), horizon_hour=0, rows_per_site=10, rng=rng, error_scale=0.3
    )
    many = make_results(
        pv_ids=range(100, 140), horizon_hour=1, rows_per_site=10, rng=rng, error_scale=0.3
    )
    results_df = pd.concat([few, many], ignore_index=True)
    out = metrics(results_df, make_metadata(results_df)).set_index("horizon_group")

    width_few = out.loc["0-0", "mae_ci_high"] - out.loc["0-0", "mae_ci_low"]
    width_many = out.loc["1-1", "mae_ci_high"] - out.loc["1-1", "mae_ci_low"]
    assert width_few > width_many
    assert out.loc["0-0", "n"] == 4 * 10
    assert out.loc["1-1", "n"] == 40 * 10


def test_bootstrap_ci_brackets_the_mae_and_is_deterministic():
    rng = np.random.default_rng(2)
    error_df = pd.DataFrame(
        {
            "pv_id": np.repeat(np.arange(20), 15),
            "abs_error": np.abs(rng.normal(0.3, 0.15, size=300)),
        }
    )
    low, high = bootstrap_mae_ci(error_df)
    assert low < error_df["abs_error"].mean() < high
    # seeded resampling: identical call gives identical interval
    assert (low, high) == bootstrap_mae_ci(error_df)


def test_single_site_gives_nan_interval():
    error_df = pd.DataFrame({"pv_id": [1] * 10, "abs_error": np.linspace(0, 1, 10)})
    low, high = bootstrap_mae_ci(error_df)
    assert np.isnan(low) and np.isnan(high)


def test_night_rows_are_filtered_by_default():
    rng = np.random.default_rng(3)
    results_df = make_results(pv_ids=range(3), horizon_hour=1, rows_per_site=5, rng=rng, error=0.1)
    results_df["generation_power"] = 0.05  # night-level output everywhere
    out = metrics(results_df, make_metadata(results_df))
    assert len(out) == 0
