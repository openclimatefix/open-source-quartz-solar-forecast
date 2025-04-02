import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from psp.data_sources.nwp import NwpDataSource
from psp.models.recent_history import compute_history_per_horizon
from psp.serialization import load_model
from psp.typings import Horizons, X


def test_compute_history_per_horizon():
    raw_data = [
        #
        [dt.datetime(2000, 1, 1, 1), 2],
        [dt.datetime(2000, 1, 1, 5), 3],
        [dt.datetime(2000, 1, 1, 9), 4],
        [dt.datetime(2000, 1, 1, 11), 5],
        [dt.datetime(2000, 1, 1, 13), 6],
        #
        [dt.datetime(2000, 1, 2, 1), 5],
        [dt.datetime(2000, 1, 2, 4, 30), 12],
        [dt.datetime(2000, 1, 2, 5), 6],
        [dt.datetime(2000, 1, 2, 9), 7],
        [dt.datetime(2000, 1, 2, 10, 30), 13],
        #
        [dt.datetime(2000, 1, 3, 1), 8],
        [dt.datetime(2000, 1, 3, 5), 9],
        [dt.datetime(2000, 1, 3, 9), 10],
    ]

    dates, values = zip(*raw_data)

    array = xr.DataArray(list(values), coords={"ts": list(dates)}, dims=["ts"])

    now = dt.datetime(2000, 1, 3, 2, 30)

    # 7 to get one horizon in the second day
    horizons = Horizons(duration=4 * 60, num_horizons=7)

    history = compute_history_per_horizon(
        array,
        now=now,
        horizons=horizons,
    )

    # columns = day0 = 2000-1-1, day1= 2000-1-2
    # rows = horizon 0, horizon 1, ...
    expected_history = np.array(
        [
            # 2h30 - 6h30
            [np.nan, 3, 9],
            # 6h30 - 10h30
            [np.nan, 4, 7],
            # 10h30 - 14h30
            [np.nan, 5.5, 13],
            # 14h30 - 18h30
            [np.nan, np.nan, np.nan],
            # 18h30 - 22h30
            [np.nan, np.nan, np.nan],
            # 22h30 - 2h30
            [2, 5, 8],
            # 2h30 again
            [np.nan, 3, 9],
        ]
    )

    assert_array_equal(expected_history, history)


def _predict(pv_data_source, nwp_data_source, ts=dt.datetime(2020, 1, 10), pv_id="8229"):
    """Predict for given data sources.

    Common code for the test_predict_* tests.
    """
    # We load some model, it could be any model really.
    model = load_model("psp/tests/fixtures/models/model_v8.pkl")

    model.set_data_sources(
        pv_data_source=pv_data_source,
        nwp_data_sources=nwp_data_source,
    )

    return model.predict(X(ts=ts, pv_id=pv_id))


def test_predict_with_missing_features(pv_data_source, nwp_data_sources):
    """Test that if we `predict` with missing features results in an error."""
    nwp_data_sources["UKV"]._data = nwp_data_sources["UKV"]._data.drop_isel(variable=0)

    with pytest.raises(RuntimeError) as e:
        _predict(pv_data_source, nwp_data_sources)
    assert "was trained on features" in str(e.value)


def test_predict_with_extra_features(pv_data_source, nwp_data_source):
    """Test that if we `predict` with extra features results in an error."""
    nwp_data_source._data = nwp_data_source._data.drop_isel(variable=0)
    nwp_data = nwp_data_source._data

    # Add an extra variable
    var_d = nwp_data.sel(variable="hcc")
    var_d.coords["variable"] = "patate"
    nwp_data = xr.concat([nwp_data, var_d], dim="variable")
    nwp_data_sources = {"UKV": NwpDataSource(nwp_data)}

    with pytest.raises(RuntimeError) as e:
        _predict(pv_data_source, nwp_data_sources)
    assert "was trained on features" in str(e.value)


@pytest.mark.parametrize("reindex", [True, False])
def test_predict_with_features_in_wrong_order(pv_data_source, nwp_data_source, reindex):
    """Test that swapping features around doesn't change anything (as long as their names remain
    the same).
    """
    # We test many combinaisons because sometimes we randomly get the same output (e.g. if the NWP
    # variables are the same).
    all_close = True
    for ts in pd.date_range(dt.datetime(2020, 1, 6, 10), dt.datetime(2020, 1, 10, 10), freq="24h"):
        # # This ensures the nwp fixture passed for the test is a dictionary
        # if isinstance(self._nwp_data_sources, dict):
        #     pass
        # else:
        #     self._nwp_data_sources = dict(nwp_data_source = self._nwp_data_sources)

        for pv_id in ["8215", "8229"]:
            y1 = _predict(pv_data_source, nwp_data_source, ts=ts, pv_id=pv_id)

            variables = list(nwp_data_source._data.coords["variable"].values)
            reversed_variables = variables[::-1]

            # Swap the NWP variables around.
            if reindex:
                # This swaps the variable names but also the values, i.e. everything should be fine.
                nwp_data_source._data = nwp_data_source._data.reindex(
                    {"variable": reversed_variables}
                )

            else:
                # This only swaps the names but not the values so we will get a different output
                # value.
                # Note that you could be unlucky and have this do nothing if for instance the model
                # doesn't use those features.
                nwp_data_source._data = nwp_data_source._data.assign_coords(
                    variable=("variable", reversed_variables)
                )

            y2 = _predict(pv_data_source, nwp_data_source, ts=ts, pv_id=pv_id)

            if abs(y1.powers - y2.powers).mean() > 1e-6:
                all_close = False
                break

        if not all_close:
            break

    if reindex:
        # When we reindex (move the variable names AND the values), everything should be fine.
        assert all_close
    else:
        # When we don't properly reindex, and only mix the variable names, then we should get
        # different output.
        assert not all_close
