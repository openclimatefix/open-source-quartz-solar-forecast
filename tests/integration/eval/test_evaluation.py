from quartz_solar_forecast.evaluation import run_eval
import tempfile
import pandas as pd
import pytest


@pytest.mark.integration
def test_run_eval():

    # create a fake dataframe

    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)

        test_dataset = pd.DataFrame(
            columns=[
                "pv_id",
                "timestamp",
            ],
            data=[[7593, "2021-08-21 12:00:00"], [7593, "2021-10-04 20:00:00"]],
        )

        testset_filename = tmpdirname + "/test_dataset.csv"
        test_dataset.to_csv(testset_filename, index=False)

        # call the metrics function
        run_eval(testset_filename)
