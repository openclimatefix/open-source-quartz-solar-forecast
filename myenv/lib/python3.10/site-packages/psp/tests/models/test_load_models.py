import pathlib

import pytest
from numpy.testing import assert_allclose

from psp.scripts.train_model import main as train_model_main
from psp.serialization import load_model
from psp.testing import run_click_command
from psp.typings import X

# We ran the test once to get the results and pasted here the results.
# This way we can make sure the output of previous models doesn't change.
EXPECTED_OUTPUT = {
    "model_v8": [77.446404, 78.650178, 79.028029, 96.746425, 95.17781],
    "model_v9": [67.522277, 68.571797, 68.901229, 78.870112, 77.591338],
}


def test_old_models_sanity_check():
    """We simply make sure that we are testing all the models from the fixture directory."""
    models = [
        p.stem for p in pathlib.Path("psp/tests/fixtures/models").iterdir() if p.suffix == ".pkl"
    ]

    assert sorted(models) == sorted(EXPECTED_OUTPUT)


def _test_model(model_path, expected, pv_data_source, nwp_data_sources, satellite_data_source=None):
    model = load_model(model_path)
    model.set_data_sources(
        pv_data_source=pv_data_source,
        nwp_data_sources=nwp_data_sources,
        satellite_data_sources=satellite_data_source,
    )

    pv_id = pv_data_source.list_pv_ids()[0]
    max_ = pv_data_source.max_ts()
    min_ = pv_data_source.min_ts()

    timestamp = min_ + (max_ - min_) / 2

    output = model.predict(X(pv_id=pv_id, ts=timestamp))

    assert_allclose(output.powers, expected, atol=1e-6)


@pytest.mark.parametrize("model_name,expected", [("model_v8", EXPECTED_OUTPUT["model_v8"])])
def test_old_models(model_name, expected, pv_data_source, nwp_data_sources):
    """Make sure that we can load previously trained models."""
    model_path = f"psp/tests/fixtures/models/{model_name}.pkl"
    _test_model(model_path, expected, pv_data_source, nwp_data_sources)


def test_latest_model(tmp_path, pv_data_source, nwp_data_sources, satellite_data_sources):
    """Make sure that when we train a model using the "test_config1.py" we get a model
    that behaves the same as our last fixture model (see the keys of `EXPECTED_OUTPUT`).

    If this fails, it means something has changed and we want to save a new fixture
    model, trained using the "test_config1.py" config.
    """
    cmd_args = [
        "--exp-config-name",
        "test_config1",
        "--exp-root",
        str(tmp_path),
        "--exp-name",
        "train_test",
        "--batch-size",
        "1",
        "--num-test-samples",
        "0",
    ]

    run_click_command(train_model_main, cmd_args)

    # Make sure a model was created.
    _test_model(
        tmp_path / "train_test" / "model_0.pkl",
        list(EXPECTED_OUTPUT.values())[-1],
        pv_data_source,
        nwp_data_sources,
        satellite_data_sources,
    )
