import numpy as np
import pandas as pd
import pytest

from quartz_solar_forecast.eval.metrics import (
    metrics,
    mbe,
    mbe_normalized,
    rmse,
    rmse_normalized,
)


def make_eval_dfs(forecast, actual, capacity, horizon_hour):
    n = len(forecast)
    results_df = pd.DataFrame(
        {
            "pv_id": np.arange(n),
            "timestamp": pd.date_range("2024-05-01", periods=n, freq="h"),
            "horizon_hour": [horizon_hour] * n,
            "forecast_power": forecast,
            "generation_power": actual,
        }
    )
    pv_metadata = pd.DataFrame({"pv_id": np.arange(n), "capacity": capacity})
    return results_df, pv_metadata


# errors are [+1, -1, +2]: MAE = 4/3, RMSE = sqrt(2), MBE = 2/3
FORECAST = [2.0, 1.0, 5.0]
ACTUAL = [1.0, 2.0, 3.0]


def test_rmse_known_values():
    assert rmse(FORECAST, ACTUAL) == pytest.approx(np.sqrt(2.0))
    assert rmse(ACTUAL, ACTUAL) == 0.0


def test_mbe_known_values():
    assert mbe(FORECAST, ACTUAL) == pytest.approx(2.0 / 3.0)
    # symmetric errors cancel out
    assert mbe([2.0, 0.0], [1.0, 1.0]) == 0.0


def test_metrics_accept_series_and_arrays():
    forecast = pd.Series(FORECAST)
    actual = np.array(ACTUAL)
    assert rmse(forecast, actual) == pytest.approx(np.sqrt(2.0))
    assert mbe(forecast, actual) == pytest.approx(2.0 / 3.0)


def test_normalized_metrics():
    # with capacity 2 the errors halve: RMSE = sqrt(2)/2, MBE = 1/3
    assert rmse_normalized(FORECAST, ACTUAL, 2.0) == pytest.approx(np.sqrt(2.0) / 2.0)
    assert mbe_normalized(FORECAST, ACTUAL, 2.0) == pytest.approx(1.0 / 3.0)


def test_empty_inputs_return_nan():
    for func in (rmse, mbe):
        assert np.isnan(func([], []))
    for func in (rmse_normalized, mbe_normalized):
        assert np.isnan(func([], [], []))


def test_metrics_overall_values(capsys):
    results_df, pv_metadata = make_eval_dfs(FORECAST, ACTUAL, capacity=2.0, horizon_hour=3)
    metrics(results_df, pv_metadata)
    out = capsys.readouterr().out

    assert "MAE: 1.3333 kw, normalized 66.67 %" in out
    assert "RMSE: 1.4142 kw, normalized 70.71 %" in out
    assert "MBE: 0.6667 kw, normalized 33.33 %" in out


def test_metrics_per_horizon_includes_rmse_and_mbe(capsys):
    results_df, pv_metadata = make_eval_dfs(FORECAST, ACTUAL, capacity=2.0, horizon_hour=3)
    metrics(results_df, pv_metadata)
    out = capsys.readouterr().out

    # horizon [3, 3] group: RMSE = sqrt(2) ~ 1.414, MBE = 0.667
    assert "MAE for horizon" in out
    assert "rmse: 1.41 kw. mbe: 0.667 kw" in out


def test_metrics_night_filter(capsys):
    # last row is nighttime (generation 0.0) with a large forecast error
    forecast = [1.0, 2.0, 5.0, 100.0]
    actual = [2.0, 2.0, 3.0, 0.0]

    results_df, pv_metadata = make_eval_dfs(forecast, actual, capacity=1.0, horizon_hour=0)

    metrics(results_df, pv_metadata, include_night=False)
    out_night_removed = capsys.readouterr().out
    # night row dropped: errors are [-1, 0, +2] -> MAE = 1.0
    assert "MAE: 1.0 kw" in out_night_removed

    metrics(results_df, pv_metadata, include_night=True)
    out_all = capsys.readouterr().out
    # all rows kept: errors are [-1, 0, +2, +100] -> MAE = 25.75
    assert "MAE: 25.75 kw" in out_all
