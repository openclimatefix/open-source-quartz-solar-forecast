from psp.scripts.train_model import main
from psp.testing import run_click_command


def test_train_model(tmp_path):
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
        "10",
    ]

    run_click_command(main, cmd_args)

    # Make sure a model was created.
    assert (tmp_path / "train_test" / "model_0.pkl").exists()
