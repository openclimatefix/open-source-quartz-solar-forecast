import pandas as pd
import pytest

from quartz_solar_forecast.forecast import run_forecast


def test_rounding_near_midnight_tz_aware_does_not_error():
    """
    Regression test for issue #320:
    tz-aware timestamps close to midnight should not error
    and should be handled consistently.
    """
    ts = pd.Timestamp("2024-01-01 23:55", tz="Europe/London")

    # The key assertion: this should NOT raise
    result = run_forecast(ts=ts)

    # Minimal sanity check: result exists
    assert result is not None
